#include <cuda_runtime.h>
#include <cublas.h>
#include <cublas_api.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 4
//nvcc -Xcompiler -fPIC -shared -lcublas -o cuda_matmul_sgemm.so cuda_matmul_sgemm.cu

void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int n, const int k) {
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

extern "C" {
void cuda_matmul(float *a, float *b, float *c, int m, int n, int k, size_t a_size, size_t b_size, size_t c_size)
{
    float *d_a, *d_b, *d_c;

    cudaMalloc((void **)&d_a, a_size * sizeof(float));
    cudaMalloc((void **)&d_b, b_size * sizeof(float));
    cudaMalloc((void **)&d_c, c_size * sizeof(float));

    cudaMemcpy(d_a, a, a_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, b_size * sizeof(float), cudaMemcpyHostToDevice);

    gpu_blas_mmul(d_a, d_b, d_c, m, n, k);

    cudaMemcpy(c, d_c, c_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
}