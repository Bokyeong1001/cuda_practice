#include <stdio.h>
#include <stdlib.h>

__global__
void multiply(int n, float * x, float * y, float * out)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) out[i] = x[i] * y[i];
}


void fill_array(float * data, int N){
    for(int idx=0;idx<N;idx++){
        data[idx] = idx;
    }
}

void print_output(float *x, float *y, float *out, int N){
    for(int idx=0;idx<N;idx++){
        printf("%.1f X %.1f = %.1f\n", x[idx], y[idx], out[idx]);
    }
}

int main(){
    int N = 10;

    float *h_a, *h_b, *h_c;
    cudaMallocHost((void **) &h_a, sizeof(float)*N);
    cudaMallocHost((void **) &h_b, sizeof(float)*N);
    cudaMallocHost((void **) &h_c, sizeof(float)*N);

    fill_array(h_a,N);
    fill_array(h_b,N);
    
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(float)*N);
    cudaMalloc((void **) &d_b, sizeof(float)*N);
    cudaMalloc((void **) &d_c, sizeof(float)*N);

    cudaMemcpy(d_a, h_a, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float)*N, cudaMemcpyHostToDevice);

    // Perform SAXPY on 1M elements
    multiply<<<1,N>>>(N, d_a, d_b, d_c);

    cudaMemcpy(h_c, d_c, sizeof(float)*N, cudaMemcpyDeviceToHost);
    print_output(h_a, h_b, h_c, N);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    return 0;
}