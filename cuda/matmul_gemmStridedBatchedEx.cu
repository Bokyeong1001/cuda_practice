#include <cublas_v2.h>
#include <iostream>

#define M 4
#define N 4
#define K 4
//nvcc -lcublas -o matmul_gemmStridedBatchedEx matmul_gemmStridedBatchedEx.cu

void print_matrix(const float *A, int rows, int cols, int batch_count) {
    for (int k = 0; k < batch_count; k++){
        for(int j = 0; j < rows; j++){
            for(int i = 0; i < cols; i++){
                int index = i*rows + j + k*rows*cols;
                std::cout << A[index] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
}

int main()
{
    int batch_size = 4;

    float *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, sizeof(float) * M * K * batch_size);
    cudaMallocHost(&h_B, sizeof(float) * K * N * batch_size);
    cudaMallocHost(&h_C, sizeof(float) * M * N * batch_size);

    for(int k=0; k<batch_size; k++) {
        for(int j=0; j<M; j++) {
                for(int i=0; i<K; i++) {
                    int index = i*M + j + k*M*K;
                    h_A[index] = i + 0.0f;
                }       
        }
    }  

    for(int k=0; k<batch_size; k++) {
        for(int j=0; j<K; j++) {
                for(int i=0; i<N; i++) {
                    int index = i*K + j + k*K*N;
                    h_B[index] = j + 1.0f;
                }       
        }
    }

    std::cout << "A =" << std::endl;
    print_matrix(h_A, M, K, batch_size);
    std::cout << "B =" << std::endl;
    print_matrix(h_B, K, N, batch_size);

    // Allocate memory on the device
    float* d_A;
    float* d_B;
    float* d_C;
    
    cudaMalloc((void**)&d_A, M*K*batch_size*sizeof(float));
    cudaMalloc((void**)&d_B, K*N*batch_size*sizeof(float));
    cudaMalloc((void**)&d_C, M*N*batch_size*sizeof(float));

    cudaMemcpy(d_A, h_A, sizeof(float)*M * K * batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float)*K * N * batch_size, cudaMemcpyHostToDevice);

    // Set up cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set up the matrix dimensions and batch size
    int lda = M;
    int ldb = K;
    int ldc = M;

    int strideA = M*K;
    int strideB = K*N;
    int strideC = M*N;

    // Set the alpha and beta parameters for the gemm operation
    float alpha = 1.0f;
    float beta = 0.0f;

    // Perform the matrix multiplication using cublasGemmStridedBatchedEx
    cublasGemmStridedBatchedEx(handle,
                               CUBLAS_OP_N, CUBLAS_OP_N,
                               M, N, K,
                               &alpha,
                               d_A, CUDA_R_32F, lda, strideA,
                               d_B, CUDA_R_32F, ldb, strideB,
                               &beta,
                               d_C, CUDA_R_32F, ldc, strideC,
                               batch_size,
                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cudaMemcpy(h_C,d_C,sizeof(float) * M * N * batch_size,cudaMemcpyDeviceToHost);
    std::cout << "C =" << std::endl;
    print_matrix(h_C, M, N, batch_size);

    // Clean up
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
