#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <math.h>             // fabs

//nvcc -Xcompiler -fPIC -shared -lcusparse -o spmm_batched.so spmm_batched.cu

extern "C" {
void spmm_batched(float *hA_values, float *hB, float *hC, int *hA_csrOffsets, int *hA_columns, int m, int n, int k, int A_nnz, int num_batches)
{
    // Host problem definition
    int   A_num_rows  = m;
    int   A_num_cols  = k;
    int   B_num_rows  = k;
    int   B_num_cols  = n;
    int   ldb         = B_num_rows;
    int   ldc         = A_num_rows;
    int   B_size      = ldb * B_num_cols;
    int   C_size      = ldc * B_num_cols;

    float alpha            = 1.0f;
    float beta             = 0.0f;

    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dB, *dC;
    cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int));
    cudaMalloc((void**) &dA_columns,
                           A_nnz * num_batches * sizeof(int));
    cudaMalloc((void**) &dA_values,
                           A_nnz * num_batches * sizeof(float));
    cudaMalloc((void**) &dB,
                           B_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dC,
                           C_size * num_batches * sizeof(float));

    cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice);
    cudaMemcpy(dA_columns, hA_columns, A_nnz * num_batches * sizeof(int),
                           cudaMemcpyHostToDevice);
    cudaMemcpy(dA_values, hA_values, A_nnz * num_batches * sizeof(float),
                           cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, B_size * num_batches * sizeof(float),
                           cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, C_size * num_batches * sizeof(float),
                           cudaMemcpyHostToDevice);
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    cusparseCreate(&handle);
    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCsrSetStridedBatch(matA, num_batches, 0, A_nnz);
    // Alternatively, the following code can be used for matA broadcast
    // cusparseCsrSetStridedBatch(matA, num_batches, 0, 0);
    // Create dense matrix B
    cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseDnMatSetStridedBatch(matB, num_batches, B_size);
    // Create dense matrix C
    cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseDnMatSetStridedBatch(matC, num_batches, C_size);

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
    cusparseDestroy(handle);
    //--------------------------------------------------------------------------
    // device result check
    cudaMemcpy(hC, dC, C_size * num_batches * sizeof(float),
                           cudaMemcpyDeviceToHost);
    //--------------------------------------------------------------------------
    // device memory deallocation
    cudaFree(dBuffer);
    cudaFree(dA_csrOffsets);
    cudaFree(dA_columns);
    cudaFree(dA_values);
    cudaFree(dB);
    cudaFree(dC);
}
}