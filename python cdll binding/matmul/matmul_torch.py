import torch
from torch import nn
import ctypes
from ctypes import *
#import nvidia_dlprof_pytorch_nvtx
#nvidia_dlprof_pytorch_nvtx.init()


#(float *h_A, float *h_B, float *h_C, int M, int N, int K, int batch_size)
def get_cuda_matmul():
    dll = ctypes.CDLL('./cuda_matmul.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.cuda_matmul
    func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_int]
    return func

__cuda_matmul = get_cuda_matmul()

if __name__ == '__main__':
    #start = torch.cuda.Event(enable_timing=True)
    #end = torch.cuda.Event(enable_timing=True)

    m = 256
    n = 256
    k = 256
    num_batches = 256

    hA = torch.rand((m*k*num_batches), dtype=torch.float32, device='cuda')
    hB = torch.rand((k*n*num_batches), dtype=torch.float32, device='cuda')
    hC = torch.empty((m*n*num_batches), dtype=torch.float32, device='cuda')

    """for b in range(num_batches):
        for j in range(m):
            for i in range(k):
                index = i*m + j + b*m*k
                hA[index] = i%4
        for j in range(k):
            for i in range(n):
                index = i*k + j + b*k*n
                hB[index] = i%4+1"""

    #print("A = ")
    #print_matrix(hA,m,k,num_batches)
    #print("B = ")
    #print_matrix(hB,k,n,num_batches)

    hA_p = hA.contiguous().data_ptr()
    hB_p = hB.contiguous().data_ptr()
    hC_p = hC.contiguous().data_ptr()

    #with torch.autograd.profiler.emit_nvtx():
        #start.record()
    __cuda_matmul(hA_p, hB_p, hC_p, m, n, k, num_batches)
        #end.record()
    #torch.cuda.synchronize()
    

    #print("C = ")
    #print_matrix(hC,m,n,num_batches)
    #print(start.elapsed_time(end))
