import cupy as cp
import numpy as np
import time
start = time.time()
saxpy_kernel = cp.RawKernel(r'''
extern "C" __global__
void saxpy(float a, float *x, float *y, float *out, size_t n)
{
 size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
 if (tid < n) {
   out[tid] = a * x[tid] + y[tid];
  }
 }
''', 'saxpy')

NUM_THREADS = 128  # Threads per block
NUM_BLOCKS = 8  # Blocks per grid
a = cp.array([2.0], dtype=cp.float32)
n = NUM_THREADS * NUM_BLOCKS

hX = cp.random.rand(n).astype(dtype=cp.float32)
hY = cp.random.rand(n).astype(dtype=cp.float32)
hOut = cp.empty(n).astype(dtype=cp.float32)

kernel_start = time.time()
saxpy_kernel((NUM_BLOCKS,), (NUM_THREADS,), (a, hX, hY, hOut, n))  # grid, block and arguments
end = time.time()
#print(hOut)
print('kernel time elapsed:', end - kernel_start)
print('program time elapsed:', end - start)
#kernel time elapsed: 0.0004904270172119141
#program time elapsed: 0.38382458686828613