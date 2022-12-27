#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

//nvcc -lcusparse -o sddmm_batched sddmm_batched.cu
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {
 
    for(int i = 0; i < nr_rows_A; i++){
        for(int j = 0; j < nr_cols_A; j++){
            printf("%0.1f ",A[i * nr_cols_A + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_matrixC(const float *C, const int *hC_offsets, const int *hC_columns, int nr_rows_C, int nr_cols_C) {
    int k = 0;
    for(int i = 1; i < nr_rows_C+1; i++){
        for(int j = 0; j < nr_cols_C; j++){
            if(j==hC_columns[k]){
                if(k<hC_offsets[i]){
                    printf("%0.1f ", C[k]);
                    k += 1;
                }
                else{
                    printf("0 ");
                }
            }
            else{
                printf("0 ");
            }
        }
        printf("\n");
    }
    printf("\n");
}

int main(void) {
    // Host problem definition
    int   A_num_rows   = 4;
    int   A_num_cols   = 4;
    int   B_num_rows   = A_num_cols;
    int   B_num_cols   = 3;
    int   C_nnz        = 9;
    int   lda          = A_num_cols;
    int   ldb          = B_num_cols;
    int   A_size       = lda * A_num_rows;
    int   B_size       = ldb * B_num_rows;
    int   num_batches  = 2;

    float hA1[]        = { 1.0f,   2.0f,  3.0f,  4.0f,
                           5.0f,   6.0f,  7.0f,  8.0f,
                           9.0f,  10.0f, 11.0f, 12.0f,
                           13.0f, 14.0f, 15.0f, 16.0f };
    float hA2[]        = { 10.0f,   11.0f,  12.0f,  13.0f,
                           14.0f,   15.0f,  16.0f,  17.0f,
                           18.0f,   19.0f,  20.0f,  21.0f,
                           22.0f,   23.0f,  24.0f,  25.0f };
    float hB1[]        = {  1.0f,  2.0f,  3.0f,
                            4.0f,  5.0f,  6.0f,
                            7.0f,  8.0f,  9.0f,
                            10.0f, 11.0f, 12.0f };
    float hB2[]        = {  6.0f,  4.0f,  2.0f,
                            3.0f,  7.0f,  1.0f,
                            9.0f,  5.0f,  2.0f,
                            8.0f,  4.0f,  7.0f };
    int   hC_offsets[]  = { 0, 3, 4, 7, 9 };
    int   hC_columns1[] = { 0, 1, 2, 1, 0, 1, 2, 0, 2 };
    int   hC_columns2[] = { 0, 1, 2, 0, 0, 1, 2, 1, 2 };
    float hC_values1[]  = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                           0.0f, 0.0f, 0.0f, 0.0f };
    float hC_values2[]  = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                           0.0f, 0.0f, 0.0f, 0.0f };
    float hC_result1[]  = { 70.0f, 80.0f, 90.0f,
                           184.0f,
                           246.0f, 288.0f, 330.0f,
                           334.0f, 450.0f };
    float hC_result2[]  = {305.0f, 229.0f, 146.0f,
                           409.0f,
                           513.0f, 389.0f, 242.0f,
                           469.0f, 290.0f };
    float alpha        = 1.0f;
    float beta         = 0.0f;
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dC_offsets, *dC_columns;
    float *dC_values, *dB, *dA;
    CHECK_CUDA( cudaMalloc((void**) &dA,
                           A_size * num_batches * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dB,
                           B_size * num_batches * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_offsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_columns,
                           C_nnz * num_batches * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dC_values,
                           C_nnz * num_batches * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA, hA1, A_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA + A_size, hA2, A_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB1, B_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB + B_size, hB2, B_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_offsets, hC_offsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_columns, hC_columns1, C_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_columns + C_nnz, hC_columns2, C_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_values, hC_values1, C_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_values + C_nnz, hC_values2, C_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseDnMatDescr_t matA, matB;
    cusparseSpMatDescr_t matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create dense matrix A
    CHECK_CUSPARSE( cusparseCreateDnMat(&matA, A_num_rows, A_num_cols, lda, dA,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    CHECK_CUSPARSE( cusparseDnMatSetStridedBatch(matA, num_batches, A_size) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    CHECK_CUSPARSE( cusparseDnMatSetStridedBatch(matB, num_batches, B_size) )
    // Create sparse matrix C in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, A_num_rows, B_num_cols, C_nnz,
                                      dC_offsets, dC_columns, dC_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCsrSetStridedBatch(matC, num_batches, 0, C_nnz) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSDDMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute preprocess (optional)
    CHECK_CUSPARSE( cusparseSDDMM_preprocess(
                                  handle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                  CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer) )
    // execute SpMM
    CHECK_CUSPARSE( cusparseSDDMM(handle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                  CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer) )
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroyDnMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hC_values1, dC_values, C_nnz * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC_values2, dC_values + C_nnz, C_nnz * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    int correct = 1;
    for (int i = 0; i < C_nnz; i++) {
        if (hC_values1[i] != hC_result1[i]) {
            correct = 0; // direct floating point comparison is not reliable
            break;
        }
        if (hC_values2[i] != hC_result2[i]) {
            correct = 0; // direct floating point comparison is not reliable
            break;
        }
    }
    if (correct)
        printf("sddmm_csr_batched_example test PASSED\n");
    else
        printf("sddmm_csr_batched_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC_offsets) )
    CHECK_CUDA( cudaFree(dC_columns) )
    CHECK_CUDA( cudaFree(dC_values) )
    return EXIT_SUCCESS;
}