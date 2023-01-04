#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

//nvcc -Xcompiler -fPIC -shared -lcusparse -o attn_forward.so attn_forward.cu

__global__ void softmax_kernel(float *d_out, float *d_values, int rows, int cols, int block_size, int num_batch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows*num_batch)
        return;

    float max = 0.0;
    for (int k = 0; k < block_size; k++){
        if (max < d_values[i * block_size + k]){
            max = d_values[i * block_size + k];
        }
    }
    float sum = 0.0f;
    for (int k = 0; k < cols; k++){
        if (k < block_size){
            sum += expf(d_values[i * block_size + k] - max);
        }
        else{
            sum += expf(0 - max);
        }
    }
    for (int k = 0; k < block_size; k++){
        d_out[i * block_size + k] = expf(d_values[i * block_size + k] - max) / sum;
    }
}

void spmm(cusparseHandle_t handle, void *dBuffer, float *dAttn, float *dValue, float *dOut, int *d_offsets, int *d_columns, int seq_len, int emb_dim, int nnz, int num_batches)
{
    // Host problem definition
    int   ldb         = seq_len;
    int   ldc         = seq_len;
    int   output_size = seq_len * emb_dim;

    float alpha            = 1.0f;
    float beta             = 0.0f;

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    size_t               bufferSize = 0;

    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, seq_len, seq_len, nnz,
                                      d_offsets, d_columns, dAttn,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCsrSetStridedBatch(matA, num_batches, 0, nnz);
    // Alternatively, the following code can be used for matA broadcast
    // cusparseCsrSetStridedBatch(matA, num_batches, 0, 0);
    // Create dense matrix B
    cusparseCreateDnMat(&matB, seq_len, emb_dim, ldb, dValue,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseDnMatSetStridedBatch(matB, num_batches, output_size);
    // Create dense matrix C
    cusparseCreateDnMat(&matC, seq_len, emb_dim, ldc, dOut,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseDnMatSetStridedBatch(matC, num_batches, output_size);

    // allocate an external buffer if needed
    cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG2, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // execute SpMM
    cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG2, dBuffer);

    // destroy matrix/vector descriptors
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cudaFree(dBuffer);
}

void sddmm(cusparseHandle_t handle, void *dBuffer, float *dQuery, float *dKey, float *dAttn, int *d_offsets, int *d_columns, int seq_len, int emb_dim, int nnz, int num_batches){
    
    size_t bufferSize  = 0;
    int   lda          = emb_dim;
    int   ldb          = seq_len;
    int   input_size       = seq_len * emb_dim;

    float alpha            = 1.0f;
    float beta             = 0.0f;

    cusparseDnMatDescr_t matA, matB;
    cusparseSpMatDescr_t matC;
    
    // Create dense matrix A
    cusparseCreateDnMat(&matA, seq_len, emb_dim, lda, dQuery,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseDnMatSetStridedBatch(matA, num_batches, input_size);
    // Create dense matrix B
    cusparseCreateDnMat(&matB, emb_dim, seq_len, ldb, dKey,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseDnMatSetStridedBatch(matB, num_batches, input_size);
    // Create sparse matrix C in CSR format
    cusparseCreateCsr(&matC, seq_len, seq_len, nnz,
                                      d_offsets, d_columns, dAttn,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCsrSetStridedBatch(matC, num_batches, 0, nnz);
    // allocate an external buffer if needed
    cusparseSDDMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // execute preprocess (optional)
    cusparseSDDMM_preprocess(
                                  handle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                  CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer);
    // execute SpMM
    cusparseSDDMM(handle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                  CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer);
    // destroy matrix/vector descriptors
    cusparseDestroyDnMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroySpMat(matC);
    cudaFree(dBuffer);
}

extern "C" {
void attn_forward(float *hQuery, float *hKey, float *hValue, float *hAttn, float *hAttnOut, float *hOut, int *hOffsets, int *hColumns, int seq_len, int emb_dim, int nnz, int block_size, int num_batches)
{
    // Host problem definition
    int   input_size       = seq_len * emb_dim;

    int   *d_offsets, *d_columns;
    float *dQuery, *dKey, *dValue, *dAttn, *dAttnOut, *dOut;

    cudaMalloc((void**) &dQuery, input_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dKey, input_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dValue, input_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dAttn, nnz * num_batches * sizeof(float));
    cudaMalloc((void**) &dAttnOut, nnz * num_batches * sizeof(float));
    cudaMalloc((void**) &d_offsets, (seq_len + 1) * sizeof(int));
    cudaMalloc((void**) &d_columns, nnz * num_batches * sizeof(int));
    cudaMalloc((void**) &dOut, input_size * num_batches * sizeof(float));

    cudaMemcpy(dQuery, hQuery, input_size * sizeof(float) * num_batches, cudaMemcpyHostToDevice);
    cudaMemcpy(dKey, hKey, input_size * sizeof(float) * num_batches, cudaMemcpyHostToDevice);
    cudaMemcpy(dValue, hValue, input_size * sizeof(float) * num_batches, cudaMemcpyHostToDevice);
    cudaMemcpy(dAttn, hAttn, nnz * sizeof(float) * num_batches, cudaMemcpyHostToDevice);
    cudaMemcpy(dAttnOut, hAttnOut, nnz * sizeof(float) * num_batches, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, hOffsets, (seq_len + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, hColumns, nnz * sizeof(int) * num_batches, cudaMemcpyHostToDevice);
    cudaMemcpy(dOut, hOut, input_size * sizeof(float) * num_batches, cudaMemcpyHostToDevice);

    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseCreate(&handle);
    void* dBuffer    = NULL;

    sddmm(handle, dBuffer, dQuery, dKey, dAttn, d_offsets, d_columns, seq_len, emb_dim, nnz, num_batches);
    cudaMemcpy(hAttn, dAttn, nnz * sizeof(float) * num_batches, cudaMemcpyDeviceToHost);

    softmax_kernel<<<num_batches, seq_len>>>(dAttnOut, dAttn, seq_len, seq_len, block_size, num_batches);
    cudaMemcpy(hAttnOut, dAttnOut, nnz * sizeof(float) * num_batches, cudaMemcpyDeviceToHost);

    spmm(handle, dBuffer, dAttnOut, dValue, dOut, d_offsets, d_columns, seq_len, emb_dim, nnz, num_batches);
    cudaMemcpy(hOut, dOut, input_size * sizeof(float) * num_batches, cudaMemcpyDeviceToHost);
    //--------------------------------------------------------------------------
    // device memory deallocation
    
    cudaFree(dQuery);
    cudaFree(dKey);
    cudaFree(dValue);
    cudaFree(dAttn);
    cudaFree(dAttnOut);
    cudaFree(dOut);
    cudaFree(d_offsets);
    cudaFree(d_columns);
    
    cusparseDestroy(handle);
}
}
