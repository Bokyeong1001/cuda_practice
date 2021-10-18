import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

mod = SourceModule("""
  __global__ void print_from_gpu()
  {
     printf("Hello World! from thread {%d,%d}\
        From device\\n", blockIdx.x,threadIdx.x);
  }
  """)

func = mod.get_function("print_from_gpu")
func(block=(128, 1, 1),grid=(8,1))
