import torch
from torch import nn
import ctypes
from ctypes import *
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

def print_matrix_row(matrix,row,col,num_batches):
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

def make_mask(offsets,columns,row,col,cnt,num_batches):
    mask = torch.zeros((num_batches, seq_len, seq_len), dtype=torch.float32, device='cuda')
    k = 0
    for b in range(num_batches):
        k = 0
        for j in range(1,row+1):
            for i in range(col):
                if(i==columns[k+cnt*b]):
                    if(k<offsets[j]):
                        mask[b][j-1][i] = 1
                        k += 1
                    else:
                        mask[b][j-1][i] = 0
                else:
                    mask[b][j-1][i] = 0
    return mask

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

#(float *hQuery, float *hKey, float *hValue, float *hAttn, float *hAttnOut, float *hOut, int *hOffsets, int *hColumns, int seq_len, int emb_dim, int nnz, int block_size, int num_batches)

def get_attn_forward():
    dll = ctypes.CDLL('./block_attn_forward.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.attn_forward
    func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_int, c_int, c_int, c_int, c_int]
    return func

__cuda_attn_forward = get_attn_forward()

if __name__ == '__main__':
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    seq_len = 4
    emb_dim = 4

    num_batches = 1
    block_size = 2
    hOffsets = [0]
    offset = 0

    for _ in range(seq_len):
        offset += block_size
        hOffsets.append(offset)
    
    hColumns = []

    for i in range(int(seq_len/block_size)):
        column = block_size*i
        for _ in range(block_size):
            column = block_size*i
            for _ in range(block_size):
                hColumns.append(column)
                column += 1

    hColumns = hColumns * num_batches
    nnz = hOffsets[-1]
    #print(hC_offsets)
    #print(hC_columns)
    #print(C_nnz)

    hQuery = torch.rand((seq_len*emb_dim*num_batches), dtype=torch.float32, device='cuda')
    hKey = torch.rand((seq_len*emb_dim*num_batches), dtype=torch.float32, device='cuda')
    hValue = torch.rand((seq_len*emb_dim*num_batches), dtype=torch.float32, device='cuda')
    hAttn = torch.zeros(num_batches*nnz, dtype=torch.float32, device='cuda')
    hAttnOut = torch.zeros(num_batches*nnz, dtype=torch.float32, device='cuda')
    hOut = torch.zeros(seq_len*emb_dim*num_batches, dtype=torch.float32, device='cuda')
    hOffsets = torch.tensor(hOffsets,dtype=torch.int32, device='cuda')
    hColumns = torch.tensor(hColumns,dtype=torch.int32, device='cuda')

    """for b in range(num_batches):
        for j in range(seq_len):
            for i in range(emb_dim):
                index = i*seq_len + j + b*seq_len*emb_dim
                hQuery[index] = i%4
                hKey[index] = i%4
                hValue[index] = i%4"""


    print("Query = ")
    print_matrix(hQuery, seq_len, emb_dim, num_batches)
    print("Key = ")
    print_matrix(hKey, seq_len, emb_dim, num_batches)
    print("Value = ")
    print_matrix_row(hValue, seq_len, emb_dim, num_batches)

    hQuery_p = hQuery.contiguous().data_ptr()
    hKey_p = hKey.contiguous().data_ptr()
    hValue_p = hValue.contiguous().data_ptr()
    hAttn_p = hAttn.contiguous().data_ptr()
    hAttnOut_p = hAttnOut.contiguous().data_ptr()
    hOut_p = hOut.contiguous().data_ptr()
    offsets_p = hOffsets.contiguous().data_ptr()
    columns_p = hColumns.contiguous().data_ptr()

    #with torch.autograd.profiler.emit_nvtx():
    __cuda_attn_forward(hQuery_p, hKey_p, hValue_p, hAttn_p, hAttnOut_p, hOut_p, offsets_p, columns_p, seq_len, emb_dim, nnz, block_size, num_batches)
    
    print("Attn = ")
    print_sparse_matrix(hAttn,hOffsets,hColumns, seq_len, seq_len, nnz, num_batches)
    torch_q = hQuery.view(num_batches, seq_len, emb_dim)
    torch_k = hKey.view(num_batches, emb_dim, seq_len)
    torch_v = hValue.view(num_batches, seq_len, emb_dim)
    mask = make_mask(hOffsets,hColumns,seq_len,seq_len,nnz,num_batches)
    torch_attn = torch.matmul(torch_q, torch_k) * mask
    print("Torch Attn = ")
    print(torch_attn)

    print("Attn Out = ")
    print_sparse_matrix(hAttnOut,hOffsets,hColumns, seq_len, seq_len, nnz, num_batches)

    print("Torch Attn Out = ")
    softmax = torch.nn.Softmax(dim=-1)
    torch_attn_out = softmax(torch_attn/math.sqrt(emb_dim)) * mask
    print(torch_attn_out)
    
    print("\nOut = ")
    print_matrix_row(hOut, seq_len, emb_dim, num_batches)
    torch_v = torch_v.permute(0,2,1)
    torch_out = torch.matmul(torch_attn_out, torch_v)
    print("Torch Out = ")
    print(torch_out)
