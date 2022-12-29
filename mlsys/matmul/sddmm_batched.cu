#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

//nvcc -Xcompiler -fPIC -shared -lcusparse -o sddmm_batched.so sddmm_batched.cu

extern "C" {
void sddmm_batched(float *hA, float *hB, float *hC_values, int *hC_offsets, int *hC_columns, int m, int n, int k, int C_nnz, int num_batches)
{
    // Host problem definition
    int   A_num_rows   = m;
    int   A_num_cols   = k;
    int   B_num_rows   = A_num_cols;
    int   B_num_cols   = n;
    int   lda          = A_num_cols;
    int   ldb          = B_num_cols;
    int   A_size       = lda * A_num_rows;
    int   B_size       = ldb * B_num_rows;

    float alpha        = 1.0f;
    float beta         = 0.0f;

    int   *dC_offsets, *dC_columns;
    float *dC_values, *dB, *dA;

    cudaMalloc((void**) &dA, A_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dB, B_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dC_offsets, (A_num_rows + 1) * sizeof(int));
    cudaMalloc((void**) &dC_columns, C_nnz * num_batches * sizeof(int));
    cudaMalloc((void**) &dC_values, C_nnz * num_batches * sizeof(float));

    cudaMemcpy(dA, hA, A_size * sizeof(float) * num_batches, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, B_size * sizeof(float) * num_batches, cudaMemcpyHostToDevice);
    cudaMemcpy(dC_offsets, hC_offsets, (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dC_columns, hC_columns, C_nnz * sizeof(int) * num_batches, cudaMemcpyHostToDevice);
    cudaMemcpy(dC_values, hC_values, C_nnz * sizeof(float) * num_batches, cudaMemcpyHostToDevice);

    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseDnMatDescr_t matA, matB;
    cusparseSpMatDescr_t matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    cusparseCreate(&handle);
    // Create dense matrix A
    cusparseCreateDnMat(&matA, A_num_rows, A_num_cols, lda, dA,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseDnMatSetStridedBatch(matA, num_batches, A_size);
    // Create dense matrix B
    cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseDnMatSetStridedBatch(matB, num_batches, B_size);
    // Create sparse matrix C in CSR format
    cusparseCreateCsr(&matC, A_num_rows, B_num_cols, C_nnz,
                                      dC_offsets, dC_columns, dC_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCsrSetStridedBatch(matC, num_batches, 0, C_nnz);
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
    cusparseDestroy(handle);

    cudaMemcpy(hC_values, dC_values, C_nnz * sizeof(float) * num_batches,
                           cudaMemcpyDeviceToHost);

    //--------------------------------------------------------------------------
    // device memory deallocation
    cudaFree(dBuffer);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC_offsets);
    cudaFree(dC_columns);
    cudaFree(dC_values);
}
}
