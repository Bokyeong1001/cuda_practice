#include <stdio.h>
#include <math.h>

#define ROWS 2
#define COLS 10
//nvcc -o softmax_mat softmax_mat.cu

__global__ void softmax_base(float *d_out, float *d_in, int rows, int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows)
        return;

    float sum = 0.0f;
    for (int k = 0; k < cols; k++)
        sum += expf(d_in[i * cols + k]);
    for (int j = 0; j < cols; j++)
        d_out[i * cols + j] = expf(d_in[i * cols + j]) / sum;
}

int main()
{
    float h_in[ROWS*COLS] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                              0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    float h_out[ROWS*COLS];

    float *d_in;
    float *d_out;
    cudaMalloc((void **)&d_in, ROWS * COLS * sizeof(float));
    cudaMalloc((void **)&d_out, ROWS * COLS * sizeof(float));

    cudaMemcpy(d_in, h_in, ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);
    softmax_base<<<1, ROWS>>>(d_out, d_in, ROWS, COLS);
    cudaMemcpy(h_out, d_out, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++)
            printf("%f ", h_out[i * COLS + j]);
        printf("\n");
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
