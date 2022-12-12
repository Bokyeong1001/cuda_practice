from __future__ import division
from numba import cuda
import numpy as np
import math

# CUDA kernel
@cuda.jit
def matmul(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)

    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp
        
# Host code

# Initialize the data arrays
N = 4
h_A = np.zeros((N, N),dtype=np.int32)
h_B = np.zeros((N, N),dtype=np.int32)

for i in range(N):
    for j in range(N):
        h_A[i][j] = (j%4) + 1
        h_B[i][j] = (j%4) + 5

# Copy the arrays to the device
A_global_mem = cuda.to_device(h_A)
B_global_mem = cuda.to_device(h_B)

# Allocate memory on the device for the result
C_global_mem = cuda.device_array((N, N))

# Configure the blocks
threadsperblock = (N, N)
blockspergrid_x = int(math.ceil(h_A.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(h_B.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Start the kernel 
matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)

# Copy the result back to the host
h_C = C_global_mem.copy_to_host()

print(h_C)
print(np.matmul(h_A,h_B))