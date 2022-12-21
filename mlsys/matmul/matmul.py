import numpy as np
import ctypes
from ctypes import *

# extract cuda_sum function pointer in the shared object cuda_sum.so
def get_cuda_matmul():
    dll = ctypes.CDLL('./cuda_matmul_cublas.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.cuda_matmul
    func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_size_t]
    return func

# create __cuda_sum function with get_cuda_sum()
__cuda_matmul = get_cuda_matmul()

# convenient python wrapper for __cuda_sum
# it does all job with types convertation
# from python ones to C++ ones
def cuda_matmul(a, b, c, N, size):
    a_p = a.ctypes.data_as(POINTER(c_float))
    b_p = b.ctypes.data_as(POINTER(c_float))
    c_p = c.ctypes.data_as(POINTER(c_float))

    __cuda_matmul(a_p, b_p, c_p, N, size)

# testing, sum of two arrays of ones and output head part of resulting array
if __name__ == '__main__':
    size=int(4*4)

    a = np.ones(size).astype('float32')
    b = np.ones(size).astype('float32')
    c = np.zeros(size).astype('float32')
    N = 4

    cuda_matmul(a, b, c, N, size)

    print(c)