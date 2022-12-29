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

void print_matrix(const float *A, int nr_rows_A, int nr_cols_A, int batch_size) {
    for (int k = 0; k < batch_size; k++){
        for(int j = 0; j < nr_rows_A; j++){
            for(int i = 0; i < nr_cols_A; i++){
                int index = j*nr_cols_A + i + k*nr_rows_A*nr_cols_A;
                printf("%0.1f ",A[index]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void print_matrixC(const float *C, const int *hC_offsets, const int *hC_columns, int nr_rows_C, int nr_cols_C, int C_nnz, int batch_size) {
    int k = 0;
    for(int b = 0; b < batch_size; b++){
        printf("\n%d\n", k);
        k=0;
        for(int i = 1; i < nr_rows_C+1; i++){
            for(int j = 0; j < nr_cols_C; j++){
                if(j==hC_columns[k+C_nnz*b]){
                    if(k<hC_offsets[i]){
                        printf("%0.1f ", C[k+C_nnz*b]);
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

    int   hC_offsets[]  = { 0, 3, 4, 7, 9 };
    int   hC_columns[] = { 0, 1, 2, 1, 0, 1, 2, 0, 2, 
                           0, 1, 2, 1, 0, 1, 2, 0, 2};
    float alpha        = 1.0f;
    float beta         = 0.0f;

    float *hA, *hB, *hC_values;
    cudaMallocHost(&hA, sizeof(float) * A_num_rows * A_num_cols * num_batches);
    cudaMallocHost(&hB, sizeof(float) * B_num_rows * B_num_cols * num_batches);
    cudaMallocHost(&hC_values, sizeof(float) * C_nnz * num_batches);

    for(int k=0; k<num_batches; k++) {
        for(int j=0; j<A_num_rows; j++) {
            for(int i=0; i<A_num_cols; i++) {
                int index = j*A_num_cols + i + k*A_num_rows*A_num_cols;
                hA[index] = i+j + 0.0f;
            }       
        }
    }  

    for(int k=0; k<num_batches; k++) {
        for(int j=0; j<B_num_rows; j++) {
            for(int i=0; i<B_num_cols; i++) {
                int index = j*B_num_cols + i + k*B_num_rows*B_num_cols;
                hB[index] = i+j + 0.0f;
            }       
        }
    }
    printf("A = \n");
    print_matrix(hA, A_num_rows, A_num_cols,num_batches);

    printf("B = \n");
    print_matrix(hB, B_num_rows, B_num_cols,num_batches);

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

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size * sizeof(float) * num_batches,
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(float) * num_batches,
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_offsets, hC_offsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_columns, hC_columns, C_nnz * sizeof(int) * num_batches,
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_values, hC_values, C_nnz * sizeof(float) * num_batches,
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
    CHECK_CUDA( cudaMemcpy(hC_values, dC_values, C_nnz * sizeof(float) * num_batches,
                           cudaMemcpyDeviceToHost) )
    printf("C = \n");
    /*for (int i = 0; i < C_nnz; i++) {
        printf("%0.1f ", hC_values[i]);
    }*/
    print_matrixC(hC_values, hC_offsets, hC_columns, A_num_rows, B_num_cols, C_nnz, num_batches);

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