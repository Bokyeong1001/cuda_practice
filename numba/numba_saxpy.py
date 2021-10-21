from __future__ import print_function
import sys
import numpy as np
from numba import cuda, float32, void
import time
start = time.time()
# GPU code
# ---------

@cuda.jit(void(float32, float32[:], float32[:], float32[:]))
def saxpy(a, x, y, out):
	i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
	# Map i to array elements
	if i >= out.size:
		# Out of range?
		return
	# Do actual work
	out[i] = a * x[i] + y[i]

# CPU code
# ---------

NUM_THREADS = 128  # Threads per block
NUM_BLOCKS = 8  # Blocks per grid 
N = NUM_BLOCKS * NUM_THREADS

a = np.float32(2.)				# Force value to be float32

# Generate numbers 0..(NELEM - 1)
x = np.arange(N, dtype='float32')
y = np.arange(N, dtype='float32')
out = np.empty_like(x)
kernel_start = time.time()
saxpy[NUM_BLOCKS, NUM_THREADS](a, x, y, out)
end = time.time()
#print("out =", out)
print('kernel time elapsed:', end - kernel_start)
print('program time elapsed:', end - start)
#kernel time elapsed: 0.0016009807586669922
#program time elapsed: 0.24187493324279785

