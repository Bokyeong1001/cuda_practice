#include <iostream>
#include <cuda_runtime.h>
#include <cublas.h>
#include <cublas_api.h>
#include <cublas_v2.h>

using namespace std;
//nvcc -lcublas -o matmul_gemmEX_tensor matmul_gemmEX_tensor.cu

void blas_forward(const float *A, const float *B, float *C, const int m, const int n, const int k) {
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
 
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
 
    // Do the actual multiplication
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb, beta, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    // Destroy the handle
    cublasDestroy(handle);
}


void print_matrix(float *A, int rows, int cols, int batch_size) {
    for (int i = 0; i < batch_size; i++){
        for (int j = 0; j < rows; j++){
            for(int k = 0; k < cols; k++){
                std::cout << A[i*rows*cols+k * rows + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

}

int main() {
     // Allocate 3 arrays on CPU
     int batch_size = 2; 
     int M = 2; 
     int K = 3;
     int N = 4;
 
 
     float *h_A, *h_B, *h_C;
     cudaMallocHost((void **) &h_A, sizeof(float) * M * K * batch_size);
     cudaMallocHost((void **) &h_B, sizeof(float) * K * N * batch_size);
     cudaMallocHost((void **) &h_C, sizeof(float) * M * N * batch_size);
     
     for (int i = 0; i < batch_size; i++){
        for (int j = 0; j < M * K; j++)
            h_A[i*M*K+j] = j;
        for (int j = 0; j < K * N; j++)
            h_B[i*K*N+j] = j + 4;
        for (int j = 0; j < M * N; j++)
            h_C[i*M*N+j] = 0;
     }
     
     // Allocate 3 arrays on GPU
     float *d_A, *d_B, *d_C;
     cudaMalloc((void **) &d_A, sizeof(float)*M * K * batch_size);
     cudaMalloc((void **) &d_B, sizeof(float)*K * N * batch_size);
     cudaMalloc((void **) &d_C, sizeof(float)*M * N * batch_size);
 
     // copy matrix A and B from host to device memory
     cudaMemcpy(d_A, h_A, sizeof(float)*M * K * batch_size, cudaMemcpyHostToDevice);
     cudaMemcpy(d_B, h_B, sizeof(float)*K * N * batch_size, cudaMemcpyHostToDevice);

     std::cout << "A =" << std::endl;
     print_matrix(h_A, M, K, batch_size);

     std::cout << "B =" << std::endl;
     print_matrix(h_B, K, N, batch_size);
 
     // Multiply A and B on GPU
     blas_forward(d_A, d_B, d_C, M, N, K);
 
     // Copy (and print) the result on host memory
     cudaMemcpy(h_C,d_C,M * N * sizeof(float) * batch_size,cudaMemcpyDeviceToHost);
     std::cout << "C =" << std::endl;
     print_matrix(h_C, M, N, batch_size);

     //Free GPU memory
     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);
 
     // Free CPU memory
     cudaFreeHost(h_A);
     cudaFreeHost(h_B);
     cudaFreeHost(h_C);
 
     return 0;
 }