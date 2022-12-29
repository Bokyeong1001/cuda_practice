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
                index = j*col + i + b*row*col
                print(matrix[index].item(), end=' ')
            print()
        print()


def print_matrixC(matrix,offsets,columns,row,col,cnt,num_batches):
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

#(float *hA, float *hB, float *hC_values, int *hC_offsets, int *hC_columns, int m, int n, int k, int C_nnz, int num_batches)
def get_sddmm_batched():
    dll = ctypes.CDLL('./sddmm_batched.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.sddmm_batched
    func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_int, c_int, c_int, c_int, c_int]
    return func

__cuda_sddmm = get_sddmm_batched()

if __name__ == '__main__':
    #start = torch.cuda.Event(enable_timing=True)
    #end = torch.cuda.Event(enable_timing=True)

    m = 16
    n = 16
    k = 16
    num_batches = 1
    block_size = 4
    hC_offsets = [0]
    offset = 0
    for _ in range(m):
        offset += block_size
        hC_offsets.append(offset)
    
    hC_columns = []

    for i in range(int(m/block_size)):
        column = block_size*i
        for _ in range(block_size):
            column = block_size*i
            for _ in range(block_size):
                hC_columns.append(column)
                column += 1

    hC_columns = hC_columns * num_batches
    C_nnz = hC_offsets[-1]
    #print(hC_offsets)
    #print(hC_columns)
    #print(C_nnz)

    hA = torch.zeros((m*k*num_batches), dtype=torch.float32, device='cuda')
    hB = torch.zeros((k*n*num_batches), dtype=torch.float32, device='cuda')
    hC_values = torch.zeros(num_batches*C_nnz, dtype=torch.float32, device='cuda')
    hC_offsets = torch.tensor(hC_offsets,dtype=torch.int32, device='cuda')
    hC_columns = torch.tensor(hC_columns,dtype=torch.int32, device='cuda')

    for b in range(num_batches):
        for j in range(m):
            for i in range(k):
                index = j*k + i + b*m*k
                hA[index] = i%4
        for j in range(k):
            for i in range(n):
                index = j*n + i + b*k*n
                hB[index] = i%4+1

    print("A = ")
    print_matrix(hA,m,k,num_batches)
    print("B = ")
    print_matrix(hB,k,n,num_batches)

    hA_p = hA.contiguous().data_ptr()
    hB_p = hB.contiguous().data_ptr()
    hC_p = hC_values.contiguous().data_ptr()
    offsets_p = hC_offsets.contiguous().data_ptr()
    columns_p = hC_columns.contiguous().data_ptr()

    #with torch.autograd.profiler.emit_nvtx():
        #start.record()
    __cuda_sddmm(hA_p, hB_p, hC_p, offsets_p, columns_p, m, n, k, C_nnz, num_batches)
        #end.record()
    #torch.cuda.synchronize()
    

    print("C = ")
    print_matrixC(hC_values,hC_offsets,hC_columns,m,n,C_nnz,num_batches)
    #print(start.elapsed_time(end))
