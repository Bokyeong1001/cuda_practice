#include <cuda_runtime_api.h> 
#include <cublas_v2.h>

#include <math.h>

//nvcc -Xcompiler -fPIC -shared -lcublas -o attn_backward.so attn_backward.cu

__global__ void softmax_scale_backward_kernel(float *dGradAttnScore, float *dAttnScore, float *dGradAttnScoreScale, float *dGradSum, float *dGradAttnScale,
                                            float *dGradAttn, int seq_len, int emb_dim, int num_batches)
{
    float scale = sqrtf(float(emb_dim));
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= seq_len*num_batches)
        return;

    for (int k = 0; k < seq_len; k++){
        dGradAttnScoreScale[k * seq_len + i] = dGradAttnScore[k * seq_len + i] * dAttnScore[i * seq_len + k];
    }

    for (int k = 0; k < seq_len; k++){
        dGradSum[i] += dGradAttnScoreScale[k * seq_len + i];

    }
    for (int k = 0; k < seq_len; k++){
        dGradAttn[k * seq_len + i] = (dGradAttnScoreScale[k * seq_len + i] - (dAttnScore[i * seq_len + k] * dGradSum[i]))/scale;
    }
}

void softmax_scale_backward_function(float *dGradAttnScore, float *dAttnScore, float *dGradAttnScoreScale, float *dGradSum, float *dGradAttnScale,
                                            float *dGradAttn, int seq_len, int emb_dim, int num_batches)
{
    softmax_scale_backward_kernel<<<num_batches, seq_len>>>(dGradAttnScore, dAttnScore, dGradAttnScoreScale, dGradSum, dGradAttnScale, dGradAttn, seq_len, emb_dim, num_batches);
}

void gemm_strided_batchedEx(cublasHandle_t handle, cublasOperation_t opA, const float *d_A, const float *d_B, float *d_C, const int M, const int N, const int K, const int batch_size) {

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
                               opA, CUBLAS_OP_N,
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
void attn_backward(float *hQuery, float *hKey, float *hValue, float *hAttnScore, float *hGradOutput, float *hGradAttnScore, float *hGradAttnScoreScale, 
                    float *hGradSum, float *hGradAttnScale, float *hGradAttn, float *hGradQuery, float *hGradKey, float *hGradValue,
                    int seq_len, int emb_dim, int num_batches)
{
    int input_size = seq_len * emb_dim;
    int attn_size = seq_len * seq_len;

    float *dQuery, *dKey, *dValue, *dAttnScore, *dGradOutput, *dGradAttnScore, *dGradAttnScoreScale, *dGradSum, *dGradAttnScale, *dGradAttn, *dGradQuery, *dGradKey, *dGradValue;

    cudaMalloc((void**) &dQuery, input_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dKey, input_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dValue, input_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dAttnScore, attn_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dGradOutput, input_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dGradAttnScore, attn_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dGradAttnScoreScale, attn_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dGradSum, seq_len * num_batches * sizeof(float));
    cudaMalloc((void**) &dGradAttnScale, attn_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dGradAttn, attn_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dGradQuery, input_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dGradKey, input_size * num_batches * sizeof(float));
    cudaMalloc((void**) &dGradValue, input_size * num_batches * sizeof(float));

    cudaMemcpy(dQuery, hQuery, input_size * sizeof(float) * num_batches, cudaMemcpyHostToDevice);
    cudaMemcpy(dKey, hKey, input_size * num_batches * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dValue, hValue, input_size * num_batches * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dAttnScore, hAttnScore, attn_size * num_batches * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dGradAttn, hGradAttn, attn_size * num_batches * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dGradOutput, hGradOutput, input_size * num_batches * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    gemm_strided_batchedEx(handle, CUBLAS_OP_N, dGradOutput, dValue, dGradAttnScore, seq_len, emb_dim, seq_len, num_batches);
    cudaMemcpy(hGradAttnScore, dGradAttnScore, attn_size * sizeof(float) * num_batches, cudaMemcpyDeviceToHost);

    softmax_scale_backward_function(dGradAttnScore, dAttnScore, dGradAttnScoreScale, dGradSum, dGradAttnScale, dGradAttn, seq_len, emb_dim, num_batches);
    cudaMemcpy(hGradAttnScoreScale, dGradAttnScoreScale, attn_size * sizeof(float) * num_batches, cudaMemcpyDeviceToHost);
    cudaMemcpy(hGradSum, dGradSum, seq_len * sizeof(float) * num_batches, cudaMemcpyDeviceToHost);
    cudaMemcpy(hGradAttn, dGradAttn, attn_size * sizeof(float) * num_batches, cudaMemcpyDeviceToHost);

    gemm_strided_batchedEx(handle, CUBLAS_OP_T, dGradAttn, dKey, dGradQuery, seq_len, seq_len, seq_len, num_batches);
    cudaMemcpy(hGradQuery, dGradQuery, input_size * sizeof(float) * num_batches, cudaMemcpyDeviceToHost);

    gemm_strided_batchedEx(handle, CUBLAS_OP_N, dGradAttn, dQuery, dGradKey, seq_len, seq_len, seq_len, num_batches);
    cudaMemcpy(hGradKey, dGradKey, input_size * sizeof(float) * num_batches, cudaMemcpyDeviceToHost);

    gemm_strided_batchedEx(handle, CUBLAS_OP_N, dAttnScore, dGradOutput, dGradValue, seq_len, seq_len, seq_len, num_batches);
    cudaMemcpy(hGradValue, dGradValue, input_size * sizeof(float) * num_batches, cudaMemcpyDeviceToHost);


    cudaFree(dQuery);
    cudaFree(dKey);
    cudaFree(dValue);
    cudaFree(dAttnScore);
    cudaFree(dGradOutput);
    cudaFree(dGradAttnScore);
    cudaFree(dGradAttnScoreScale);
    cudaFree(dGradSum);
    cudaFree(dGradAttnScale);
    cudaFree(dGradAttn);
    cudaFree(dGradQuery);
    cudaFree(dGradKey);
    cudaFree(dGradValue);
    
    cublasDestroy(handle);
}
}
