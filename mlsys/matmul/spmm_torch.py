import torch
from torch import nn
import ctypes
from ctypes import *
import nvidia_dlprof_pytorch_nvtx
nvidia_dlprof_pytorch_nvtx.init()

def print_matrix(matrix,row,col,num_batches):
    for b in range(num_batches):
        for j in range(row):
            for i in range(col):
                index = i*row + j + b*row*col
                print(matrix[index].item(), end=' ')
            print()
        print()


def print_sparse_matrix(matrix,offsets,columns,row,col,cnt,num_batches):
    k = 0
    for b in range(num_batches):
        k = 0
        for j in range(1,row+1):
            for i in range(col):
                if(i==columns[k+cnt*b]):
                    if(k<offsets[j]):
                        print(matrix[k+cnt*b].item(), end=' ')
                        k += 1
                    else:
                        print("0.0", end=' ')
                else:
                    print("0.0", end=' ')
            print()
        print()

#(float *hA_values, float *hB, float *hC, int *hA_offsets, int *hA_columns, int m, int n, int k, int C_nnz, int num_batches)
def get_spmm_batched():
    dll = ctypes.CDLL('./spmm_batched.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.spmm_batched
    func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_int, c_int, c_int, c_int, c_int]
    return func

__cuda_spmm = get_spmm_batched()

if __name__ == '__main__':
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    m = 4
    n = 4
    k = 4
    num_batches = 1
    block_size = 2
    hA_offsets = [0]
    offset = 0
    for _ in range(m):
        offset += block_size
        hA_offsets.append(offset)
    
    hA_columns = []

    for i in range(int(m/block_size)):
        column = block_size*i
        for _ in range(block_size):
            column = block_size*i
            for _ in range(block_size):
                hA_columns.append(column)
                column += 1

    hA_columns = hA_columns * num_batches
    A_nnz = hA_offsets[-1]
    #print(hA_offsets)
    #print(hA_columns)
    #print(C_nnz)

    #hA_values = torch.zeros((A_nnz*num_batches), dtype=torch.float32, device='cuda')
    hA_values = torch.tensor([0,  0,  6,  6, 12, 12, 18, 18], dtype=torch.float32, device='cuda')
    hB = torch.zeros((k*n*num_batches), dtype=torch.float32, device='cuda')
    hC = torch.zeros(num_batches*m*n, dtype=torch.float32, device='cuda')
    hA_offsets = torch.tensor(hA_offsets,dtype=torch.int32, device='cuda')
    hA_columns = torch.tensor(hA_columns,dtype=torch.int32, device='cuda')

    for b in range(num_batches):
        for j in range(k):
            for i in range(n):
                index = i*k + j + b*k*n
                hB[index] = i%4
    
    print("A = ")
    print_sparse_matrix(hA_values,hA_offsets,hA_columns,m,n,A_nnz,num_batches)
    print("B = ")
    print_matrix(hB,k,n,num_batches)

    hA_p = hA_values.contiguous().data_ptr()
    hB_p = hB.contiguous().data_ptr()
    hC_p = hC.contiguous().data_ptr()
    offsets_p = hA_offsets.contiguous().data_ptr()
    columns_p = hA_columns.contiguous().data_ptr()

    #with torch.autograd.profiler.emit_nvtx():
    start.record()
    __cuda_spmm(hA_p, hB_p, hC_p, offsets_p, columns_p, m, n, k, A_nnz, num_batches)
    end.record()
    torch.cuda.synchronize()
    
    print("C = ")
    print_matrix(hC,m,n,num_batches)
    torch_A = hA.view(num_batches, m, k)
    
    print(start.elapsed_time(end))
