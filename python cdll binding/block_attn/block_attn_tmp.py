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

#(float *hQuery, float *hKey, float *hValue, float *hAttn, float *hOut, int *hOffsets, int *hColumns, int seq_len, int emb_dim, int nnz, int num_batches)

def get_attn():
    dll = ctypes.CDLL('./attn.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.attn
    func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_int, c_int, c_int, c_int]
    return func

__cuda_sddmm = get_attn()

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

    hQuery = torch.zeros((seq_len*emb_dim*num_batches), dtype=torch.float32, device='cuda')
    hKey = torch.zeros((seq_len*emb_dim*num_batches), dtype=torch.float32, device='cuda')
    hValue = torch.zeros((seq_len*emb_dim*num_batches), dtype=torch.float32, device='cuda')
    hAttn = torch.zeros(num_batches*nnz, dtype=torch.float32, device='cuda')
    hOut = torch.zeros(seq_len*emb_dim*num_batches, dtype=torch.float32, device='cuda')
    hOffsets = torch.tensor(hOffsets,dtype=torch.int32, device='cuda')
    hColumns = torch.tensor(hColumns,dtype=torch.int32, device='cuda')

    for b in range(num_batches):
        for j in range(seq_len):
            for i in range(emb_dim):
                index = i*seq_len + j + b*seq_len*emb_dim
                hQuery[index] = i%4
                hKey[index] = i%4
                hValue[index] = i%4


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
    hOut_p = hOut.contiguous().data_ptr()
    offsets_p = hOffsets.contiguous().data_ptr()
    columns_p = hColumns.contiguous().data_ptr()

    #with torch.autograd.profiler.emit_nvtx():
    start.record()
    __cuda_sddmm(hQuery_p, hKey_p, hValue_p, hAttn_p, hOut_p, offsets_p, columns_p, seq_len, emb_dim, nnz, num_batches)
    end.record()
    torch.cuda.synchronize()
    
    print("Attn = ")
    print_sparse_matrix(hAttn,hOffsets,hColumns, seq_len, seq_len, nnz, num_batches)
    torch_q = hQuery.view(num_batches, seq_len, emb_dim)
    torch_k = hKey.view(num_batches, emb_dim, seq_len)
    torch_v = hValue.view(num_batches, seq_len, emb_dim)
    mask = make_mask(hOffsets,hColumns,seq_len,seq_len,nnz,num_batches)
    torch_attn = torch.matmul(torch_q, torch_k) * mask
    print("Torch Attn = ")
    print(torch_attn)

    print("\nOut = ")
    print_matrix(hOut, seq_len, emb_dim, num_batches)
    torch_v = torch_v.permute(0,2,1)
    torch_out = torch.matmul(torch_attn, torch_v)
    print("Torch Out = ")
    print(torch_out)
    print(start.elapsed_time(end))
