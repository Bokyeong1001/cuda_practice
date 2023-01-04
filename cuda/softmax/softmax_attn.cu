#include <stdio.h>
#include <math.h>

#define ROWS 6
#define COLS 6
#define BLOCK_SIZE 2
//nvcc -o softmax_attn softmax_attn.cu

__global__ void softmax(float *d_out, float *d_values, int *d_columns, int *d_csrOffsets, int rows, int cols, int block_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows)
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

int main()
{
    //float h_values[ROWS*BLOCK_SIZE] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f };
    float *h_values;
    int   h_csrOffsets[]  = { 0, 2, 4, 6, 8, 10, 12 };
    int   h_columns[]    = { 0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5 };
    float h_out[ROWS*BLOCK_SIZE];

    cudaMallocHost(&h_values, sizeof(float) * ROWS * BLOCK_SIZE);
    for(int j = 0; j < ROWS * BLOCK_SIZE; j++) {
        h_values[j] = j+1; 
    }

    float *d_values;
    int *d_csrOffsets;
    int *d_columns;
    float *d_out;

    cudaMalloc((void **)&d_values, ROWS * BLOCK_SIZE * sizeof(float));
    cudaMalloc((void **)&d_columns, ROWS * BLOCK_SIZE * sizeof(int));
    cudaMalloc((void **)&d_csrOffsets, (ROWS + 1) * sizeof(int));
    cudaMalloc((void **)&d_out, ROWS * BLOCK_SIZE * sizeof(float));

    cudaMemcpy(d_values, h_values, ROWS * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, h_columns, ROWS * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrOffsets, h_csrOffsets, (ROWS + 1) * BLOCK_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    softmax<<<1, ROWS>>>(d_out, d_values, d_columns, d_csrOffsets, ROWS, COLS, BLOCK_SIZE);

    cudaMemcpy(h_out, d_out, ROWS * BLOCK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++)
            printf("%f ", h_out[i * BLOCK_SIZE + j]);
        printf("\n");
    }

    cudaFree(d_values);
    cudaFree(d_out);
    cudaFree(d_csrOffsets);
    cudaFree(d_columns);

    return 0;
}
