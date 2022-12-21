#include <iostream>
#include <cuda_runtime.h>
#include <cublas.h>
#include <cublas_api.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 4

using namespace std;
//nvcc -lcublas -o matmul_sgemm matmul_sgemm.cu

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

void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
     int lda=m,ldb=k,ldc=m;
     const float alf = 1;
     const float bet = 0;
     const float *alpha = &alf;
     const float *beta = &bet;
 
     // Create a handle for CUBLAS
     cublasHandle_t handle;
     cublasCreate(&handle);
 
     // Do the actual multiplication
     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
 
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
     int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
 
     // for simplicity we are going to use square arrays
     nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 3;
 
     float *h_A, *h_B, *h_C;
     cudaMallocHost((void **) &h_A, sizeof(float) * nr_rows_A * nr_cols_A);
     cudaMallocHost((void **) &h_B, sizeof(float) * nr_rows_B * nr_cols_B);
     cudaMallocHost((void **) &h_C, sizeof(float) * nr_rows_C * nr_cols_C);
     
     int N = 3;
     for (int i = 0; i < N; ++i) {
         for (int j = 0; j < N; ++j) {
             h_A[i * N + j] =  i+j;
         }
     }
 
     // random initialize matrix B
     for (int i = 0; i < N; ++i) {
         for (int j = 0; j < N; ++j) {
             h_B[i * N + j] = i+j;
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
     gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
 
     // Copy (and print) the result on host memory
     cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
     std::cout << "C =" << std::endl;
     print_matrix(h_C, nr_rows_C, nr_cols_C);
 
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