#include <stdio.h>
#include <math.h>
//nvcc -o softmax softmax.cu

__global__ void kernel(float *d_out, float *d_in, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float sum = 0.0f;
    for (int j = 0; j < n; j++)
        sum += expf(d_in[j]);
    d_out[i] = expf(d_in[i]) / sum;
}

int main()
{
    const int N = 10;
    float h_in[N] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    float h_out[N];

    float *d_in;
    float *d_out;
    cudaMalloc((void **)&d_in, N * sizeof(float));
    cudaMalloc((void **)&d_out, N * sizeof(float));

    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    kernel<<<1, N>>>(d_out, d_in, N);
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        printf("%f ", h_out[i]);

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}