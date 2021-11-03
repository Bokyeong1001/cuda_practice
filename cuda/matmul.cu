 #include <stdio.h>
 #include <stdlib.h>
 #include <assert.h>
 
 #define BLOCK_SIZE 16

 __global__ void gpu_matrix_mult(int *A,int *B, int *C, int N)
 { 
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = tmpSum;
 } 

 void cpu_matrix_mult(int *h_a, int *h_b, int *h_cc, int N) {
     for (int i = 0; i < N; ++i) 
     {
         for (int j = 0; j < N; ++j) 
         {
             int tmp = 0.0;
             for (int h = 0; h < N; ++h) 
             {
                 tmp += h_a[i * N + h] * h_b[h * N + j];
             }
             h_cc[i * N + j] = tmp;
         }
     }
 }

 int main(int argc, char const *argv[])
 {
    int N = 16;
    int SIZE = N*N;

     /* Fixed seed for illustration */
     srand(3333);
 
     // allocate memory in host RAM, h_cc is used to store CPU result
     int *h_a, *h_b, *h_c, *h_cc;
     cudaMallocHost((void **) &h_a, sizeof(int)*SIZE);
     cudaMallocHost((void **) &h_b, sizeof(int)*SIZE);
     cudaMallocHost((void **) &h_c, sizeof(int)*SIZE);
     cudaMallocHost((void **) &h_cc, sizeof(int)*SIZE);
 
     // random initialize matrix A
     for (int i = 0; i < N; ++i) {
         for (int j = 0; j < N; ++j) {
             h_a[i * N + j] = rand() % 1024;
         }
     }
 
     // random initialize matrix B
     for (int i = 0; i < N; ++i) {
         for (int j = 0; j < N; ++j) {
             h_b[i * N + j] = rand() % 1024;
         }
     }

     // Allocate memory space on the device 
     int *d_a, *d_b, *d_c;
     cudaMalloc((void **) &d_a, sizeof(int)*SIZE);
     cudaMalloc((void **) &d_b, sizeof(int)*SIZE);
     cudaMalloc((void **) &d_c, sizeof(int)*SIZE);
 
     // copy matrix A and B from host to device memory
     cudaMemcpy(d_a, h_a, sizeof(int)*SIZE, cudaMemcpyHostToDevice);
     cudaMemcpy(d_b, h_b, sizeof(int)*SIZE, cudaMemcpyHostToDevice);
 
     unsigned int grid_rows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
     unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
     dim3 dimGrid(grid_cols, grid_rows);
     dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    
     gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);    
    
     // Transefr results from device to host 
     cudaMemcpy(h_c, d_c, sizeof(int)*SIZE, cudaMemcpyDeviceToHost);
     cudaDeviceSynchronize();
 
     cpu_matrix_mult(h_a, h_b, h_cc, N);
 
     // validate results computed by GPU
     int all_ok = 1;
     for (int i = 0; i < N; ++i)
     {
         for (int j = 0; j < N; ++j)
         {
             //printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_cc[i*k + j], i, j, h_c[i*k + j]);
             if(h_cc[i*N + j] != h_c[i*N + j])
             {
                 all_ok = 0;
             }
         }
         //printf("\n");
     }
 
     if(all_ok)
     {
        printf("all results are correct!\n");
     }
     else
     {
        printf("incorrect results\n");
     }
 
     // free memory
     cudaFree(d_a);
     cudaFree(d_b);
     cudaFree(d_c);
     cudaFreeHost(h_a);
     cudaFreeHost(h_b);
     cudaFreeHost(h_c);
     cudaFreeHost(h_cc);
     return 0;
 }