#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <math.h>             // fabs

//nvcc -lcusparse -o spmm_batched spmm_batched.cu

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
                int index = i*nr_rows_A + j + k*nr_rows_A*nr_cols_A;
                printf("%0.1f ",A[index]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void print_sparse_matrix(const float *values, const int *offsets, const int *columns, int rows, int cols, int nnz, int batch_size) {
    int k = 0;
    for(int b = 0; b < batch_size; b++){
        k=0;
        for(int i = 1; i < rows+1; i++){
            for(int j = 0; j < cols; j++){
                if(j==columns[k+nnz*b]){
                    if(k<offsets[i]){
                        printf("%0.1f ", values[k+nnz*b]);
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
    printf("\n");
}


int main(void) {
    // Host problem definition
    int   A_num_rows  = 4;
    int   A_num_cols  = 4;
    int   A_nnz       = 8;
    int   B_num_rows  = A_num_cols;
    int   B_num_cols  = 4;
    int   ldb         = B_num_rows;
    int   ldc         = A_num_rows;
    int   B_size      = ldb * B_num_cols;
    int   C_size      = ldc * B_num_cols;
    int   num_batches = 2;

    int   hA_csrOffsets[]  = { 0, 2, 4, 6, 8 };
    int   hA_columns[]    = { 0, 1, 0, 1, 2, 3, 2, 3,
                              0, 1, 0, 1, 2, 3, 2, 3 };
    float alpha            = 1.0f;
    float beta             = 0.0f;

    float *hA_values, *hB, *hC;
    cudaMallocHost(&hA_values, sizeof(float) * A_nnz  * num_batches);
    cudaMallocHost(&hB, sizeof(float) * B_num_rows * B_num_cols * num_batches);
    cudaMallocHost(&hC, sizeof(float) * A_num_rows * B_num_cols * num_batches);

    for(int k=0; k<num_batches; k++) {
        for(int j=0; j<A_nnz; j++) {
            int index = k*A_nnz + j;
            hA_values[index] = index; 
        }
    }  

    for(int k=0; k<num_batches; k++) {
        for(int j=0; j<B_num_rows; j++) {
            for(int i=0; i<B_num_cols; i++) {
                int index = i*B_num_rows + j + k*B_num_rows*B_num_cols;
                hB[index] = i+j + 0.0f;
            }       
        }
    }
    printf("A = \n");
    print_sparse_matrix(hA_values, hA_csrOffsets, hA_columns, A_num_rows, A_num_cols, A_nnz, num_batches);

    printf("B = \n");
    print_matrix(hB, B_num_rows, B_num_cols,num_batches);
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dB, *dC;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns,
                           A_nnz * num_batches * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,
                           A_nnz * num_batches * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dB,
                           B_size * num_batches * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC,
                           C_size * num_batches * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * num_batches * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * num_batches * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * num_batches * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * num_batches * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    CHECK_CUSPARSE( cusparseCsrSetStridedBatch(matA, num_batches, 0, A_nnz) )
    // Alternatively, the following code can be used for matA broadcast
    // CHECK_CUSPARSE( cusparseCsrSetStridedBatch(matA, num_batches, 0, 0) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    CHECK_CUSPARSE( cusparseDnMatSetStridedBatch(matB, num_batches, B_size) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    CHECK_CUSPARSE( cusparseDnMatSetStridedBatch(matC, num_batches, C_size) )

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG2, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG2, dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size * num_batches * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    printf("C = \n");
    print_matrix(hC, A_num_rows, B_num_cols,num_batches);
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    return EXIT_SUCCESS;
}