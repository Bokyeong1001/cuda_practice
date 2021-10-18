import cupy as cp
import numpy as np
print_from_gpu = cp.RawKernel(r'''
extern "C" __global__
void print_from_gpu(){
     printf("Hello World! from thread {%d,%d}\
        From device\n", blockIdx.x,threadIdx.x);
 }
''', 'print_from_gpu')

print_from_gpu((8,), (128,),())  # grid, block and arguments
