#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

//nvcc -lcusparse -o sddmm sddmm.cu

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
                    printf("0.0 ");
                }
            }
            else{
                printf("0.0 ");
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
    int   B_num_cols   = 4;
    int   C_nnz        = 8; //non-zero values cnt
    int   lda          = A_num_cols;
    int   ldb          = B_num_cols;
    int   A_size       = lda * A_num_rows;
    int   B_size       = ldb * B_num_rows;
    /*float hA[]         = { 1.0f,   2.0f,  3.0f,  4.0f,
                           5.0f,   6.0f,  7.0f,  8.0f,
                           9.0f,  10.0f, 11.0f, 12.0f,
                           13.0f, 14.0f, 15.0f, 16.0f };
    float hB[]         = {  1.0f,  2.0f,  3.0f,
                            4.0f,  5.0f,  6.0f,
                            7.0f,  8.0f,  9.0f,
                           10.0f, 11.0f, 12.0f };*/
    int   hC_offsets[] = { 0, 2, 4, 6, 8 }; //row별 누적값, 시작은 0. 따라서 row + 1 size
    int   hC_columns[] = { 0, 1, 0, 1, 2, 3, 2, 3}; //row별 값이 있는 col index 
    float hC_values[]  = { 0.0f, 0.0f, 0.0f, 0.0f, 
                           0.0f, 0.0f, 0.0f, 0.0f };
    float alpha        = 1.0f;
    float beta         = 0.0f;

    float *hA, *hB;
    cudaMallocHost((void **) &hA, sizeof(float) * A_num_rows * A_num_cols);
    cudaMallocHost((void **) &hB, sizeof(float) * B_num_rows * B_num_cols);

    for (int i = 0; i < A_num_rows; i++) {
        for (int j = 0; j < A_num_cols; j++) {
            hA[i * A_num_cols + j] =  i+j;
        }
    }
 
     // random initialize matrix B
    for (int i = 0; i < B_num_rows; i++) {
        for (int j = 0; j < B_num_cols; j++) {
            hB[i * B_num_cols + j] = i+j;
        }
    }

    printf("A = \n");
    print_matrix(hA, A_num_rows, A_num_cols);

    printf("B = \n");
    print_matrix(hB, B_num_rows, B_num_cols);

    //--------------------------------------------------------------------------
    // Device memory management
    int   *dC_offsets, *dC_columns;
    float *dC_values, *dB, *dA;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_offsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_columns, C_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dC_values,  C_nnz * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_offsets, hC_offsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_columns, hC_columns, C_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_values, hC_values, C_nnz * sizeof(float),
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
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create sparse matrix C in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, A_num_rows, B_num_cols, C_nnz,
                                      dC_offsets, dC_columns, dC_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
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
    CHECK_CUDA( cudaMemcpy(hC_values, dC_values, C_nnz * sizeof(float),
                           cudaMemcpyDeviceToHost) )

    printf("C = \n");
    print_matrixC(hC_values, hC_offsets, hC_columns, A_num_rows, B_num_cols);

    /*int correct = 1;
    for (int i = 0; i < C_nnz; i++) {
        if (hC_values[i] != hC_result[i]) {
            correct = 0; // direct floating point comparison is not reliable
            break;
        }
    }
    if (correct)
        printf("sddmm_csr_example test PASSED\n");
    else
        printf("sddmm_csr_example test FAILED: wrong result\n");*/
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