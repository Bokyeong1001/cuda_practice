import torch
from torch import nn
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
def cuda_matmul(a, b, c, m, n, k, a_size, b_size, c_size):
    a_p = a.contiguous().data_ptr()
    b_p = b.contiguous().data_ptr()
    c_p = c.contiguous().data_ptr()

    __cuda_matmul(a_p, b_p, c_p, m, n, k, a_size, b_size, c_size)

# testing, sum of two arrays of ones and output head part of resulting array
if __name__ == '__main__':
    #size=int(128*128)

    #N = 128
    m = 2
    n = 3
    k = 4
    input_size = m * k
    weight_size = k * n
    output_size = m * n

    """input = torch.zeros((m,k), dtype=torch.float32, device='cuda')
    for i in range(m):
        for j in range(k):
            input[i][j] = i+j
            
    weight2 = torch.ones((n,k), dtype=torch.float32, device='cuda')
    torch.manual_seed(0)
    nn.init.uniform_(weight2, -0.1, 0.1)
    print(weight2)

    #inputT = input.t().flatten()
    #weightT = weight.t().flatten()
    output = torch.zeros((m,n), dtype=torch.float32, device='cuda')"""

    grad_output = torch.ones((m,n), dtype=torch.float32, device='cuda')

    input = torch.zeros((m,k), dtype=torch.float32, device='cuda')
    for i in range(m):
        for j in range(k):
            input[i][j] = i+j
    grad_weight = torch.zeros((k,n), dtype=torch.float32, device='cuda')
    
    print(grad_output)
    print(input)
    
    with torch.autograd.profiler.emit_nvtx():
        input_tmp = input.t()
        a_p = grad_output.contiguous().data_ptr()
        b_p = input_tmp.contiguous().data_ptr()
        c_p = grad_weight.contiguous().data_ptr()
        a_size = output_size
        b_size = input_size
        c_size = weight_size

        __cuda_matmul(a_p, b_p, c_p, n, k, m, a_size, b_size, c_size)
        print(grad_weight.t())
    print(torch.matmul(grad_output.t(),input))