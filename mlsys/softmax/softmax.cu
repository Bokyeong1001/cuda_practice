#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>

// nvcc -Xcompiler -fPIC -shared -o softmax.so softmax.cu

__global__ void softmax_kernel(float *d_out, float *d_values, int rows, int cols, int block_size, int num_batch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows*num_batch)
        return;

    float sum = 0.0f;
    for (int k = 0; k < cols; k++){
        if (k < block_size){
            sum += expf(d_values[i * block_size + k]);
        }
        else{
            sum += expf(0);
        }
    }
    for (int j = 0; j < block_size; j++){
        d_out[i * block_size + j] = expf(d_values[i * block_size + j]) / sum;
    }
}

extern "C" {
void softmax(float *h_values, float *h_out, int rows, int cols, int block_size, int num_batch)
{
    float *d_values;
    float *d_out;

    cudaMalloc((void **)&d_values, rows * block_size * num_batch * sizeof(float));
    cudaMalloc((void **)&d_out, rows * block_size * num_batch * sizeof(float));

    cudaMemcpy(d_values, h_values, rows * block_size * num_batch * sizeof(float), cudaMemcpyHostToDevice);

    softmax_kernel<<<num_batch, rows>>>(d_out, d_values, rows, cols, block_size, num_batch);

    cudaMemcpy(h_out, d_out, rows * block_size * num_batch * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    cudaFree(d_out);
}
}