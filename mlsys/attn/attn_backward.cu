#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <math.h>

//nvcc -Xcompiler -fPIC -shared -lcusparse -o attn_backward.so attn_backward.cu

__global__ void softmax_scale_backward_kernel(float *dGradAttnScore, float *dAttnScore, float *dGradAttnScoreScale, float *dGradSum, float *dGradAttnScale,
                                            float *dGradAttn, int seq_len, int emb_dim, int block_size, int num_batches)
{
    float scale = sqrtf(float(emb_dim));
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len*num_batches)
        return;

    for (int k = 0; k < block_size; k++){
        dGradAttnScoreScale[i * block_size + k] = dGradAttnScore[i * block_size + k] * dAttnScore[i * block_size + k];
    }

    for (int k = 0; k < block_size; k++){
        dGradSum[i] += dGradAttnScoreScale[i * block_size + k];

    }
    for (int k = 0; k < block_size; k++){
        dGradAttn[i * block_size + k] = (dGradAttnScoreScale[i * block_size + k] - (dAttnScore[i * block_size + k] * dGradSum[i]))/scale;
    }
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

void spmm(cusparseHandle_t handle, cusparseOperation_t opA, void *dBuffer, float *dAttn, float *dValue, float *dOut, int *d_offsets, int *d_columns, int seq_len, int emb_dim, int nnz, int num_batches)
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
    cusparseSpMM_bufferSize(handle,
                            opA, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                            CUSPARSE_SPMM_CSR_ALG2, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // execute SpMM
    cusparseSpMM(handle, 
                opA, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                CUSPARSE_SPMM_CSR_ALG2, dBuffer);

    // destroy matrix/vector descriptors
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cudaFree(dBuffer);
}

extern "C" {
void attn_backward(float *hQuery, float *hKey, float *hValue, float *hAttnScore, float *hGradOutput, float *hGradAttnScore, float *hGradAttnScoreScale, 
                    float *hGradSum, float *hGradAttnScale, float *hGradAttn, float *hGradQuery, float *hGradKey, float *hGradValue,
                    int *hOffsets, int *hColumns, int seq_len, int emb_dim, int nnz, int block_size, int num_batches)
{
    int input_size = seq_len * emb_dim;

    int   *dOffsets, *dColumns;
    float *dQuery, *dKey, *dValue, *dAttnScore, *dGradOutput, *dGradAttnScore, *dGradAttnScoreScale, *dGradSum, *dGradAttnScale, *dGradAttn, *dGradQuery, *dGradKey, *dGradValue;

    cudaMalloc((void**) &dQuery, input_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dKey, input_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dValue, input_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dAttnScore, nnz * num_batches * sizeof(float));
    cudaMalloc((void**) &dGradOutput, input_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dGradAttnScore, nnz * num_batches * sizeof(float));
    cudaMalloc((void**) &dGradAttnScoreScale, nnz * num_batches * sizeof(float));
    cudaMalloc((void**) &dGradSum, seq_len * num_batches * sizeof(float));
    cudaMalloc((void**) &dGradAttnScale, nnz * num_batches * sizeof(float));
    cudaMalloc((void**) &dGradAttn, nnz * num_batches * sizeof(float));
    cudaMalloc((void**) &dGradQuery, input_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dGradKey, input_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dGradValue, input_size * num_batches * sizeof(float));

    cudaMalloc((void**) &dOffsets, (seq_len + 1) * sizeof(int));
    cudaMalloc((void**) &dColumns, nnz * num_batches * sizeof(int));

    cudaMemcpy(dQuery, hQuery, input_size * sizeof(float) * num_batches, cudaMemcpyHostToDevice);
    cudaMemcpy(dKey, hKey, input_size * num_batches * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dValue, hValue, input_size * num_batches * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dAttnScore, hAttnScore, nnz * num_batches * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dGradOutput, hGradOutput, input_size * num_batches * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(dOffsets, hOffsets, (seq_len + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dColumns, hColumns, nnz * sizeof(int) * num_batches, cudaMemcpyHostToDevice);

    cusparseHandle_t     handle = NULL;
    cusparseCreate(&handle);
    void* dBuffer    = NULL;

    sddmm(handle, dBuffer, dGradOutput, dValue, dGradAttnScore, dOffsets, dColumns, seq_len, emb_dim, nnz, num_batches);
    cudaMemcpy(hGradAttnScore, dGradAttnScore, nnz * sizeof(float) * num_batches, cudaMemcpyDeviceToHost);

    softmax_scale_backward_kernel<<<num_batches, seq_len>>>(dGradAttnScore, dAttnScore, dGradAttnScoreScale, dGradSum, dGradAttnScale, dGradAttn, seq_len, emb_dim, block_size, num_batches);
    cudaMemcpy(hGradAttnScoreScale, dGradAttnScoreScale, nnz * sizeof(float) * num_batches, cudaMemcpyDeviceToHost);
    cudaMemcpy(hGradSum, dGradSum, seq_len * sizeof(float) * num_batches, cudaMemcpyDeviceToHost);
    cudaMemcpy(hGradAttn, dGradAttn, nnz * sizeof(float) * num_batches, cudaMemcpyDeviceToHost);

    spmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, dBuffer, hGradAttn, dKey, dGradQuery, dOffsets, dColumns, seq_len, emb_dim, nnz, num_batches);
    cudaMemcpy(hGradQuery, dGradQuery, input_size * sizeof(float) * num_batches, cudaMemcpyDeviceToHost);
    
    spmm(handle, CUSPARSE_OPERATION_TRANSPOSE, dBuffer, hGradAttn, dQuery, dGradKey, dOffsets, dColumns, seq_len, emb_dim, nnz, num_batches);
    cudaMemcpy(hGradKey, dGradKey, input_size * sizeof(float) * num_batches, cudaMemcpyDeviceToHost);

    spmm(handle, CUSPARSE_OPERATION_TRANSPOSE, dBuffer, dAttnScore, dGradOutput, dGradValue, dOffsets, dColumns, seq_len, emb_dim, nnz, num_batches);
    cudaMemcpy(hGradValue, dGradValue, input_size * sizeof(float) * num_batches, cudaMemcpyDeviceToHost);

    cudaFree(dQuery);
    cudaFree(dKey);
    cudaFree(dValue);
    cudaFree(dAttnScore);
    cudaFree(dGradOutput);
    cudaFree(dGradAttnScore);
    cudaFree(dGradAttnScoreScale);
    cudaFree(dGradSum);
    cudaFree(dGradAttnScale);
    cudaFree(dGradAttn);
    cudaFree(dGradQuery);
    cudaFree(dGradKey);
    cudaFree(dGradValue);
    cudaFree(dOffsets);
    cudaFree(dColumns);
    
    cusparseDestroy(handle);
}
}
