#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublas_api.h>


using namespace std;
//nvcc -lcublas -o matmul_gemmEX matmul_gemmEX.cu

void cpu_matrix_mult(float *h_a, float *h_b, float *h_cc, int N) {
     for (int i = 0; i < N; ++i) 
     {
         for (int j = 0; j < N; ++j) 
         {
             int tmp = 0.0;
             for (int h = 0; h < N; ++h) 
             {
                 tmp += h_a[i * N + h] * h_b[h * N + j];
             }
             h_cc[i * N + j] = tmp;
         }
     }
}

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

void blas_backward(const float *A, const float *B, float *C, const int m, const int n, const int k) {
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
 
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
 
    // Do the actual multiplication
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, n, m, alpha, A, CUDA_R_32F, m, B, CUDA_R_32F, m, beta, C, CUDA_R_32F, k, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Destroy the handle
    cublasDestroy(handle);
}


void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {
 
     for(int i = 0; i < nr_rows_A; ++i){
         for(int j = 0; j < nr_cols_A; ++j){
             std::cout << A[j * nr_rows_A + i] << " ";
         }
         std::cout << std::endl;
     }
     std::cout << std::endl;
 }

int main() {
     // Allocate 3 arrays on CPU
     int nr_rows_A = 2; 
     int nr_cols_A = 3;
     int nr_rows_B = 3;
     int nr_cols_B = 4;
     int nr_rows_C = 2;
     int nr_cols_C = 4;
 
 
     float *h_A, *h_B, *h_C;
     cudaMallocHost((void **) &h_A, sizeof(float) * nr_rows_A * nr_cols_A);
     cudaMallocHost((void **) &h_B, sizeof(float) * nr_rows_B * nr_cols_B);
     cudaMallocHost((void **) &h_C, sizeof(float) * nr_rows_C * nr_cols_C);

     for (int i = 0; i < nr_rows_A; ++i) {
         for (int j = 0; j < nr_cols_A; ++j) {
             h_A[j * nr_rows_A + i] = i+j;
             //(i%nr_rows_A) +
         }
     }
 
     // random initialize matrix B
     for (int i = 0; i < nr_rows_B; ++i) {
         for (int j = 0; j < nr_cols_B; ++j) {
             h_B[j * nr_rows_B + i] = i+j;
             //(i%nr_rows_B) + 
         }
     }
     
     // Allocate 3 arrays on GPU
     float *d_A, *d_B, *d_C;
     cudaMalloc((void **) &d_A, sizeof(float)*nr_rows_A * nr_cols_A);
     cudaMalloc((void **) &d_B, sizeof(float)*nr_rows_B * nr_cols_B);
     cudaMalloc((void **) &d_C, sizeof(float)*nr_rows_C * nr_cols_C);
 
     // copy matrix A and B from host to device memory
     cudaMemcpy(d_A, h_A, sizeof(float)*nr_rows_A * nr_cols_A, cudaMemcpyHostToDevice);
     cudaMemcpy(d_B, h_B, sizeof(float)*nr_rows_B * nr_cols_B, cudaMemcpyHostToDevice);

     std::cout << "A =" << std::endl;
     print_matrix(h_A, nr_rows_A, nr_cols_A);

     std::cout << "B =" << std::endl;
     print_matrix(h_B, nr_rows_B, nr_cols_B);
 
     // Multiply A and B on GPU
     blas_forward(d_A, d_B, d_C, nr_rows_A, nr_cols_B, nr_cols_A);
 
     // Copy (and print) the result on host memory
     cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
     std::cout << "C =" << std::endl;
     print_matrix(h_C, nr_rows_C, nr_cols_C);
    
     blas_backward(d_A, d_C, d_B, nr_rows_A, nr_cols_C, nr_cols_A);
     cudaMemcpy(h_B,d_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyDeviceToHost);
     std::cout << "B =" << std::endl;
     print_matrix(h_B, nr_rows_B, nr_cols_B);

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