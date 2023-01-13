#include <iostream>
#include <cublas_v2.h>

#define M 4
#define N 4
#define K 4


//nvcc lcublas_static -o matmul_gemmBatchedEx matmul_gemmBatchedEx.cu
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

int main(int argc, char* argv[])
{
    // Linear dimension of matrices
    int batch_size = 2;

    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(M * K * sizeof(float) * batch_size);
    h_B = (float*)malloc(K * N * sizeof(float) * batch_size);
    h_C = (float*)malloc(M * N * sizeof(float) * batch_size);

    for (int i = 0; i < batch_size; i++){
        for (int j = 0; j < M * K; j++)
            h_A[i*M*K+j] = j%4;
        for (int j = 0; j < K * N; j++)
            h_B[i*K*N+j] = j%4 + 4;
        for (int j = 0; j < M * N; j++)
            h_C[i*M*N+j] = 0;
    }

    std::cout << "A =" << std::endl;
    print_matrix(h_A, M, K, batch_size);
    std::cout << "B =" << std::endl;
    print_matrix(h_B, K, N, batch_size);

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, sizeof(float)* M * K*batch_size);
    cudaMemcpy(d_A, h_A, sizeof(float)*M*K*batch_size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_B, sizeof(float)* K * N*batch_size);
    cudaMemcpy(d_B, h_B, sizeof(float)*N*K*batch_size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_C, sizeof(float)* M * N*batch_size);
    cudaMemcpy(d_C, h_C, sizeof(float)*N*M*batch_size, cudaMemcpyHostToDevice);

    float *h_dA[batch_size], *h_dB[batch_size], *h_dC[batch_size];
    for (int i = 0; i < batch_size; i++){
      h_dA[i] = d_A+i*M*K;
      h_dB[i] = d_B+i*K*N;
      h_dC[i] = d_C+i*M*N;}
      
    float **d_dA, **d_dB, **d_dC;
    cudaMalloc(&d_dA, sizeof(float *)*batch_size);
    cudaMalloc(&d_dB, sizeof(float *)*batch_size);
    cudaMalloc(&d_dC, sizeof(float *)*batch_size);
    cudaMemcpy(d_dA, h_dA, sizeof(float*)* batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dB, h_dB, sizeof(float*)* batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dC, h_dC, sizeof(float*)* batch_size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);

    // Set up the matrix dimensions and batch size
    int lda = M;
    int ldb = K;
    int ldc = M;

    // Set the alpha and beta parameters for the gemm operation
    float alpha = 1.0f;
    float beta = 0.0f;
    status = cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                            &alpha,
                            (const void**)d_dA, CUDA_R_32F, lda,
                            (const void**)d_dB, CUDA_R_32F, ldb,
                            &beta,
                            (void**)d_dC, CUDA_R_32F, ldc,
                            batch_size,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cudaMemcpy(h_C, d_C, sizeof(float)*M*N*batch_size, cudaMemcpyDeviceToHost);

    if (status == CUBLAS_STATUS_SUCCESS) {
        std::cout << "C =" << std::endl;
        print_matrix(h_C, M, N, batch_size);
    } else {
        std::cout << status << std::endl;
    }

    // Destroy the handle
    cublasDestroy(handle);


    cudaFree(d_dA);
    cudaFree(d_dB);
    cudaFree(d_dC);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}