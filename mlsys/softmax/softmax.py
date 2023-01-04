import torch
from torch import nn
import ctypes
from ctypes import *
import torch
import math
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

def python_softmax(values, rows, cols, block_size, num_batches):
    out = torch.zeros((rows*block_size*num_batches), dtype=torch.float32, device='cuda')
    for i in range(rows*num_batches):
        sum = 0
        for k in range(cols):
            if (k < block_size):
                sum += math.exp(values[i * block_size + k])
            else:
                sum += math.exp(0)
        for j in range(block_size):
            out[i * block_size + j] = math.exp(values[i * block_size + j]) / sum
    return out
    


#(float *h_values, float *h_out, int rows, int cols, int block_size, int num_batch)
def get_softmax():
    dll = ctypes.CDLL('./softmax.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.softmax
    func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, c_int, c_int, c_int, c_int]
    return func

__cuda_softmax = get_softmax()

if __name__ == '__main__':
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    rows = 6
    cols = 6
    num_batches = 1
    block_size = 2
    hA_offsets = [0]
    offset = 0
    for _ in range(rows):
        offset += block_size
        hA_offsets.append(offset)
    
    hA_columns = []

    for i in range(int(rows/block_size)):
        column = block_size*i
        for _ in range(block_size):
            column = block_size*i
            for _ in range(block_size):
                hA_columns.append(column)
                column += 1

    hA_columns = hA_columns * num_batches


    h_values = torch.zeros((rows*block_size*num_batches), dtype=torch.float32, device='cuda')
    h_out = torch.zeros((rows*block_size*num_batches), dtype=torch.float32, device='cuda')
    for j in range(rows*block_size*num_batches):
        h_values[j] = j+1; 
    
    print("A = ")
    print_sparse_matrix(h_values,hA_offsets,hA_columns,rows,cols,rows*block_size,num_batches)

    value_p = h_values.contiguous().data_ptr()
    out_p = h_out.contiguous().data_ptr()

    #with torch.autograd.profiler.emit_nvtx():
    start.record()
    __cuda_softmax(value_p, out_p, rows, cols, block_size, num_batches)
    end.record()
    torch.cuda.synchronize()
    
    print("Out = ")
    print_matrix(h_out,rows,block_size,num_batches)

    print("python Out = ")
    python_out = python_softmax(h_values,rows,cols,block_size,num_batches)
    print_matrix(python_out,rows,block_size,num_batches)

    print(start.elapsed_time(end))
