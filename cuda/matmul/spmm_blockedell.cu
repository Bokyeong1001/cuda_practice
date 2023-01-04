#include <cuda_fp16.h>        // data types
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <cstdio>            // printf
#include <cstdlib>           // EXIT_FAILURE

//nvcc -lcusparse -o spmm_blockedell spmm_blockedell.cu

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::printf("CUDA API failed at line %d with error: %s (%d)\n",        \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::printf("CUSPARSE API failed at line %d with error: %s (%d)\n",    \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {
 
    for(int i = 0; i < nr_rows_A; i++){
        for(int j = 0; j < nr_cols_A; j++){
            printf("%0.1f ",A[j * nr_rows_A + i]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_block_matrix(const float *hA_values, const int *hA_columns, int A_num_rows, int A_num_cols, int A_ell_blocksize, int A_ell_cols, int A_num_blocks) {
    int k = 0;
    int tmp = 0;
    bool block = false;
    bool init = true;
    for(int i = 0; i < A_num_rows/A_ell_blocksize; i++){
        init = true;
        for(int r=0;r<A_ell_blocksize; r++){
            if(init){
                k=A_num_blocks/A_ell_cols*i;
            }
            else{
                k=0;
            }
            for(int bc=0;bc<A_ell_cols;bc++){
                if(bc==hA_columns[k]){
                    if(k>A_num_blocks/A_ell_cols*i){
                        for(int br = 0; br < A_ell_blocksize; br++){
                            printf("0.0 ");
                        }   
                        break;
                    }
                    block = true;
                    for(int br = 0; br < A_ell_blocksize; br++){
                        printf("%.01f ", hA_values[tmp]);
                        tmp++;
                    }
                    if(block){
                        k++;
                        block=false;
                    }
                }
                else{
                    for(int br = 0; br < A_ell_blocksize; br++){
                        printf("0.0 ");
                    }
                }
            }
            printf("\n");
        }
    }
    printf("\n");
}

const int EXIT_UNSUPPORTED = 2;

int main() {
    // Host problem definition
    int   A_num_rows      = 4;
    int   A_num_cols      = 4;
    int   A_ell_blocksize = 2;
    int   A_ell_cols      = 2;
    int   A_num_blocks    = A_ell_cols * A_num_rows /
                           (A_ell_blocksize * A_ell_blocksize);
    int   B_num_rows      = A_num_cols;
    int   B_num_cols      = 3;
    int   ldb             = B_num_rows;
    int   ldc             = A_num_rows;
    int   B_size          = ldb * B_num_cols;
    int   C_size          = ldc * B_num_cols;
    int   hA_columns[]    = {1,0};
    float alpha           = 1.0f;
    float beta            = 0.0f;
    //--------------------------------------------------------------------------
    // Check compute capability
    cudaDeviceProp props;
    CHECK_CUDA( cudaGetDeviceProperties(&props, 0) )
    if (props.major < 7) {
      std::printf("cusparseSpMM with blocked ELL format is supported only "
                  "with compute capability at least 7.0\n");
      return EXIT_UNSUPPORTED;
    }

    float *hA_values, *hB, *hC;
    cudaMallocHost((void **) &hA_values, sizeof(float) * A_ell_blocksize * A_ell_blocksize * A_num_blocks);
    cudaMallocHost((void **) &hB, sizeof(float) * B_num_rows * B_num_cols);
    cudaMallocHost((void **) &hC, sizeof(float) * A_num_rows * B_num_cols);

    for (int i = 0; i < A_ell_blocksize * A_ell_blocksize * A_num_blocks; ++i) {
        hA_values[i] = i+1;
    }

    for (int i = 0; i < B_num_rows; ++i) {
        for (int j = 0; j < B_num_cols; ++j) {
            hB[j * B_num_rows + i] = j * B_num_rows + i + 1;
        }
    }

    printf("A = \n");
    print_block_matrix(hA_values, hA_columns, A_num_rows, A_num_cols, A_ell_blocksize, A_ell_cols, A_num_blocks);

    printf("B = \n");
    print_matrix(hB, B_num_rows, B_num_cols);

    //--------------------------------------------------------------------------
    // Device memory management
    int    *dA_columns;
    float *dA_values, *dB, *dC;
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_num_blocks * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,
                                    A_ell_cols * A_num_rows * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns,
                           A_num_blocks * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values,
                           A_ell_cols * A_num_rows * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in blocked ELL format
    CHECK_CUSPARSE( cusparseCreateBlockedEll(
                                      &matA,
                                      A_num_rows, A_num_cols, A_ell_blocksize,
                                      A_ell_cols, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(float),
                           cudaMemcpyDeviceToHost) )
    printf("C = \n");
    print_matrix(hC, A_num_rows, B_num_cols);

    /*int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        for (int j = 0; j < B_num_cols; j++) {
            float c_value  = static_cast<float>(hC[i + j * ldc]);
            float c_result = static_cast<float>(hC_result[i + j * ldc]);
            if (c_value != c_result) {
                correct = 0; // direct floating point comparison is not reliable
                break;
            }
        }
    }
    if (correct)
        std::printf("spmm_blockedell_example test PASSED\n");
    else
        std::printf("spmm_blockedell_example test FAILED: wrong result\n");*/
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    return EXIT_SUCCESS;
}