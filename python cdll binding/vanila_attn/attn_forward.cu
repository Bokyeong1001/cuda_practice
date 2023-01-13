#include <cuda_runtime_api.h> 
#include <cublas_v2.h>

//nvcc -Xcompiler -fPIC -shared -lcublas -o attn_forward.so attn_forward.cu

__global__ void scale_softmax_kernel(float *d_out, float *d_values, int rows, int cols, int emb_dim, int num_batch)
{
    float scale = sqrtf(float(emb_dim));
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= rows*num_batch)
        return;

    for (int k = 0; k < cols; k++){
            d_values[i * cols + k] = d_values[i * cols + k]/scale;
    }

    float max = 0.0;
    for (int k = 0; k < cols; k++){
        if (max < d_values[i * cols + k]){
            max = d_values[i * cols + k];
        }
    }
    float sum = 0.0f;
    for (int k = 0; k < cols; k++){
        sum += expf(d_values[i * cols + k] - max);
    }
    for (int k = 0; k < cols; k++){
        d_out[i * cols + k] = expf(d_values[i * cols + k] - max) / sum;
    }
}

void gemm_strided_batchedEx(cublasHandle_t handle, const float *d_A, const float *d_B, float *d_C, const int M, const int N, const int K, const int batch_size) {

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
 
    }

extern "C" {
void attn_forward(float *hQuery, float *hKey, float *hValue, float *hAttn, float *hAttnScore, float *hOut, int seq_len, int emb_dim, int num_batches)
{
    // Host problem definition
    int input_size = seq_len * emb_dim;
    int attn_size = seq_len * seq_len;

    float *dQuery, *dKey, *dValue, *dAttn, *dAttnScore, *dOut;

    cudaMalloc((void**) &dQuery, input_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dKey, input_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dValue, input_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dAttn, attn_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dAttnScore, attn_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dOut, input_size * num_batches * sizeof(float));

    cudaMemcpy(dQuery, hQuery, input_size * sizeof(float) * num_batches, cudaMemcpyHostToDevice);
    cudaMemcpy(dKey, hKey, input_size * sizeof(float) * num_batches, cudaMemcpyHostToDevice);
    cudaMemcpy(dValue, hValue, input_size * sizeof(float) * num_batches, cudaMemcpyHostToDevice);
    cudaMemcpy(dAttn, hAttn, attn_size * sizeof(float) * num_batches, cudaMemcpyHostToDevice);
    cudaMemcpy(dAttnScore, hAttnScore, attn_size * sizeof(float) * num_batches, cudaMemcpyHostToDevice);
    cudaMemcpy(dOut, hOut, input_size * sizeof(float) * num_batches, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    gemm_strided_batchedEx(handle, dQuery, dKey, dAttn, seq_len, emb_dim, seq_len, num_batches);
    cudaMemcpy(hAttn, dAttn, attn_size * sizeof(float) * num_batches, cudaMemcpyDeviceToHost);

    scale_softmax_kernel<<<num_batches, seq_len>>>(dAttnScore, dAttn, seq_len, seq_len, emb_dim, num_batches);
    cudaMemcpy(hAttnScore, dAttnScore, attn_size * sizeof(float) * num_batches, cudaMemcpyDeviceToHost);

    gemm_strided_batchedEx(handle, dAttnScore, dValue, dOut, seq_len, seq_len, seq_len, num_batches);
    cudaMemcpy(hOut, dOut, input_size * sizeof(float) * num_batches, cudaMemcpyDeviceToHost);
    
    cudaFree(dQuery);
    cudaFree(dKey);
    cudaFree(dValue);
    cudaFree(dAttn);
    cudaFree(dAttnScore);
    cudaFree(dOut);
    
    cublasDestroy(handle);
}
}
