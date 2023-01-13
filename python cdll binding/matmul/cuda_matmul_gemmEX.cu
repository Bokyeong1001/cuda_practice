#include <cuda_runtime.h>
#include <cublas.h>
#include <cublas_api.h>
#include <cublas_v2.h>

//nvcc -Xcompiler -fPIC -shared -lcublas -o cuda_matmul_gemmEX.so cuda_matmul_gemmEX.cu

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
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb, beta, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
 
    // Destroy the handle
    cublasDestroy(handle);
}

extern "C" {
void cuda_matmul(float *a, float *b, float *c, int m, int n, int k, size_t input_size, size_t weight_size, size_t output_size)
{
    float *d_a, *d_b, *d_c;

    cudaMalloc((void **)&d_a, input_size * sizeof(float));
    cudaMalloc((void **)&d_b, weight_size * sizeof(float));
    cudaMalloc((void **)&d_c, output_size * sizeof(float));

    cudaMemcpy(d_a, a, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, weight_size * sizeof(float), cudaMemcpyHostToDevice);

    gpu_blas_mmul(d_a, d_b, d_c, m, n, k);

    cudaMemcpy(c, d_c, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
}