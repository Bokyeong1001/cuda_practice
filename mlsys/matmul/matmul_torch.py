import torch
import ctypes
from ctypes import *
import nvidia_dlprof_pytorch_nvtx
nvidia_dlprof_pytorch_nvtx.init()

# extract cuda_sum function pointer in the shared object cuda_sum.so
def get_cuda_matmul():
    dll = ctypes.CDLL('./cuda_matmul_sgemm.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.cuda_matmul
    func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_int, c_int, c_int, c_size_t, c_size_t, c_size_t]
    return func

# create __cuda_sum function with get_cuda_sum()
__cuda_matmul = get_cuda_matmul()

# convenient python wrapper for __cuda_sum
# it does all job with types convertation
# from python ones to C++ ones
def cuda_matmul(a, b, c, m, n, k, input_size, weight_size, output_size):
    a_p = a.contiguous().data_ptr()
    b_p = b.contiguous().data_ptr()
    c_p = c.contiguous().data_ptr()

    __cuda_matmul(a_p, b_p, c_p, m, n, k, input_size, weight_size, output_size)

# testing, sum of two arrays of ones and output head part of resulting array
if __name__ == '__main__':
    #size=int(128*128)

    #N = 128
    m = 2
    n = 4
    k = 3
    input_size = m * k
    weight_size = k * n
    output_size = m * n

    a = torch.zeros((m,k), dtype=torch.float32, device='cuda')
    for i in range(m):
        for j in range(k):
            a[i][j] = i+j
    b = torch.zeros((k,n), dtype=torch.float32, device='cuda')
    for i in range(k):
        for j in range(n):
            b[i][j] = i+j

    aT = a.t().flatten()
    bT = b.t().flatten()
    c = torch.zeros((n,m), dtype=torch.float32, device='cuda')

    with torch.autograd.profiler.emit_nvtx():
        a_p = aT.contiguous().data_ptr()
        b_p = bT.contiguous().data_ptr()
        c_p = c.contiguous().data_ptr()

        __cuda_matmul(a_p, b_p, c_p, m, n, k, input_size, weight_size, output_size)
        print(c.t())
    print(torch.matmul(a,b))