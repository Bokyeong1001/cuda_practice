#include <cublas_v2.h>
#include <iostream>

//nvcc -Xcompiler -fPIC -shared -lcublas -o matmul_gemmStridedBatchedEx.so matmul_gemmStridedBatchedEx.cu

extern "C" {
void matmul_gemmStridedBatchedEx(float *h_A, float *h_B, float *h_C, int M, int N, int K, int batch_size)
{
    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, M*K*batch_size*sizeof(float));
    cudaMalloc((void**)&d_B, K*N*batch_size*sizeof(float));
    cudaMalloc((void**)&d_C, M*N*batch_size*sizeof(float));

    cudaMemcpy(d_A, h_A, sizeof(float)* M * K * batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float)* K * N * batch_size, cudaMemcpyHostToDevice);

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
                               CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaMemcpy(h_C,d_C,sizeof(float) * M * N * batch_size, cudaMemcpyDeviceToHost);

    // Clean up
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
}