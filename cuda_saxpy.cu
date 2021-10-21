#include <stdio.h>
#include <stdlib.h>
#define NUM_THREADS 128  
#define NUM_BLOCKS 8  

__global__
void saxpy(int n, float a, int * x, int * y, int * out)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) out[i] = a*x[i] + y[i];
}


void fill_array(int *data, int N){
    for(int idx=0;idx<N;idx++){
        data[idx] = idx;
    }
}

void print_output(int *x, int *y, int *out, int N){
    for(int idx=0;idx<N;idx++){
        printf("2 X %d + %d = %d\n", x[idx], y[idx], out[idx]);
    }
}

int main(){
    int N = NUM_BLOCKS * NUM_THREADS;
    int size = N * sizeof(int);
    int *x, *y, *out, *d_x, *d_y, *d_out;

    x = (int *)malloc(size); fill_array(x,N);
    y = (int *)malloc(size); fill_array(y,N);
    out = (int *)malloc(size);
    
    cudaMalloc(&d_x, size); 
    cudaMalloc(&d_y, size); 
    cudaMalloc(&d_out, size); 

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);

    // Perform SAXPY on 1M elements
    saxpy<<<NUM_BLOCKS,NUM_THREADS>>>(N, 2.0, d_x, d_y, d_out);

    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);
    print_output(x, y, out,N);
    free(x); free(y); free(out);
    return 0;
}