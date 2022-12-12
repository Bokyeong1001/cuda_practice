import cupy as cp
import numpy as np
import time

matmul_kernel = cp.RawKernel(r'''
extern "C" __global__ void gpu_matrix_mult(int *A, int *B, int *C, int N)
{ 
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    int tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = tmpSum;
} 
''', 'gpu_matrix_mult')

NUM_THREADS = 4  # Threads per block
NUM_BLOCKS = 1  # Blocks per grid

N = 4

h_A = cp.zeros(N*N).astype(dtype=cp.int32)
h_B = cp.zeros(N*N).astype(dtype=cp.int32)
h_C = cp.empty(N*N).astype(dtype=cp.int32)
h_Z = cp.empty(N*N).astype(dtype=cp.int32)
    
for i in range(N*N):
  h_A[i] = (i%4) + 1
  h_B[i] = (i%4) + 5


matmul_kernel((NUM_BLOCKS,NUM_BLOCKS), (NUM_THREADS,NUM_THREADS), (h_A, h_B, h_C, N))  # grid, block and arguments

for i in range(N): 
  for j in range(N):
    tmp = 0.0
    for h in range(N):
      tmp += h_A[i * N + h] * h_B[h * N + j]
    h_Z[i * N + j] = tmp

if cp.allclose(h_C, h_Z):
  print("PASS")