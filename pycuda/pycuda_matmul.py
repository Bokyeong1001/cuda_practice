import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

NUM_THREADS = 4  # Threads per block
NUM_BLOCKS = 1  # Blocks per grid

N = 4

h_A = np.zeros(N*N).astype(dtype=np.int32)
h_B = np.zeros(N*N).astype(dtype=np.int32)
h_C = np.empty(N*N).astype(dtype=np.int32)
h_Z = np.empty(N*N).astype(dtype=np.int32)
    
for i in range(N*N):
  h_A[i] = (i%4) + 1
  h_B[i] = (i%4) + 5

d_A = cuda.mem_alloc(h_A.nbytes)
d_B = cuda.mem_alloc(h_B.nbytes)
d_C = cuda.mem_alloc(h_C.nbytes)

cuda.memcpy_htod(d_A, h_A)
cuda.memcpy_htod(d_B, h_B)
cuda.memcpy_htod(d_C, h_C)

mod = SourceModule("""
  __global__ void gpu_matrix_mult(int *A, int *B, int *C, int N)
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
  """,arch="compute_80",code="sm_80")

func = mod.get_function("gpu_matrix_mult")
func(d_A,d_B,d_C,np.int32(N), block=(N,N,1), grid=(1,1))

cuda.memcpy_dtoh(h_C, d_C)
print(h_C)
