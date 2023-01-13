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

def attn_backward_torch(grad_attn_score, attn_score):
    grad_attn_score_scale = grad_attn_score * attn_score
    #print(grad_attn_score_scale) #[seq_len,seq_len] #sparse
    sum_gradient = torch.sum(grad_attn_score_scale, dim=-1, keepdim=True)
    #print(sum_gradient) #[seq_len,1]
    grad_attn_scale = grad_attn_score_scale - (attn_score * sum_gradient)
    #print(grad_attn_scale) #[seq_len,seq_len] #sparse
    grad_attn = grad_attn_scale / math.sqrt(emb_dim)

    return grad_attn

"""(float *hQuery, float *hKey, float *hValue, float *hAttnScore, float *hGradOutput, float *hGradAttnScore, float *hGradAttnScoreScale, 
    float *hGradSum, float *hGradAttnScale, float *hGradAttn, float *hGradQuery, float *hGradKey, float *hGradValue,
    int *hOffsets, int *hColumns, int seq_len, int emb_dim, int nnz, int block_size, int num_batches)"""

def get_attn_backward():
    dll = ctypes.CDLL('./attn_backward.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.attn_backward
    func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.c_void_p, ctypes.c_void_p, c_int, c_int, c_int, c_int, c_int]
    return func

__cuda_attn_backward = get_attn_backward()

if __name__ == '__main__':
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
    #print(hOffsets)
    #print(hColumns)
    #print(nnz)

    """(float *hQuery, float *hKey, float *hValue, float *hAttnScore, float *hGradOutput, float *hGradAttnScore, float *hGradAttnScoreScale, 
    float *hGradSum, float *hGradAttnScale, float *hGradAttn, float *hGradQuery, float *hGradKey, float *hGradValue,
    int *hOffsets, int *hColumns, int seq_len, int emb_dim, int nnz, int block_size, int num_batches)"""

    hQuery = torch.ones((seq_len*emb_dim*num_batches), dtype=torch.float32, device='cuda')
    hKey = torch.ones((seq_len*emb_dim*num_batches), dtype=torch.float32, device='cuda')
    hValue = torch.zeros((seq_len*emb_dim*num_batches), dtype=torch.float32, device='cuda')
    hAttnScore = torch.ones((nnz*num_batches), dtype=torch.float32, device='cuda')
    hGradOutput = torch.ones((seq_len*emb_dim*num_batches), dtype=torch.float32, device='cuda')
    hGradAttnScore = torch.zeros((nnz*num_batches), dtype=torch.float32, device='cuda')
    hGradAttnScoreScale = torch.rand((nnz*num_batches), dtype=torch.float32, device='cuda')
    hGradSum = torch.rand((seq_len*num_batches), dtype=torch.float32, device='cuda')
    hGradAttnScale = torch.rand((nnz*num_batches), dtype=torch.float32, device='cuda')
    hGradAttn = torch.zeros((nnz*num_batches), dtype=torch.float32, device='cuda')
    hGradQuery = torch.zeros((seq_len*emb_dim*num_batches), dtype=torch.float32, device='cuda')
    hGradKey = torch.zeros((seq_len*emb_dim*num_batches), dtype=torch.float32, device='cuda')
    hGradValue = torch.zeros((seq_len*emb_dim*num_batches), dtype=torch.float32, device='cuda')
    
    hOffsets = torch.tensor(hOffsets,dtype=torch.int32, device='cuda')
    hColumns = torch.tensor(hColumns,dtype=torch.int32, device='cuda')

    for i in range(nnz*num_batches):
        hAttnScore[i] = i%4+1

    for i in range(seq_len*emb_dim*num_batches):
        hValue[i] = i%4+1

    hQuery_p = hQuery.contiguous().data_ptr()
    hKey_p = hKey.contiguous().data_ptr()
    hValue_p = hValue.contiguous().data_ptr()
    hAttnScore_p = hAttnScore.contiguous().data_ptr()
    hGradOutput_p = hGradOutput.contiguous().data_ptr()
    hGradAttnScore_p = hGradAttnScore.contiguous().data_ptr()
    hGradAttnScoreScale_p = hGradAttnScoreScale.contiguous().data_ptr()
    hGradSum_p = hGradSum.contiguous().data_ptr()
    hGradAttnScale_p = hGradAttnScale.contiguous().data_ptr()
    hGradAttn_p = hGradAttn.contiguous().data_ptr()
    hGradQuery_p = hGradQuery.contiguous().data_ptr()
    hGradKey_p = hGradKey.contiguous().data_ptr()
    hGradValue_p = hGradValue.contiguous().data_ptr()
    hOffsets_p = hOffsets.contiguous().data_ptr()
    hColumns_p = hColumns.contiguous().data_ptr()

    __cuda_attn_backward(hQuery_p, hKey_p, hValue_p, hAttnScore_p, hGradOutput_p, hGradAttnScore_p, hGradAttnScoreScale_p, 
                        hGradSum_p, hGradAttnScale_p, hGradAttn_p, hGradQuery_p, hGradKey_p, hGradValue_p,
                        hOffsets_p, hColumns_p, seq_len, emb_dim, nnz, block_size, num_batches)
    
    #print_matrix(hGradOutput, seq_len, emb_dim, 1)
    #print_matrix(hValue, seq_len, emb_dim, 1)
    
    #print_sparse_matrix(hGradAttnScore, hOffsets, hColumns, seq_len, emb_dim, nnz, 1)
    #print_sparse_matrix(hGradAttnScoreScale, hOffsets, hColumns, seq_len, emb_dim, nnz, 1)
    #print(hGradSum)

    print_sparse_matrix(hGradAttn, hOffsets, hColumns, seq_len, emb_dim, nnz, 1)
    #print_matrix(hKey, seq_len, emb_dim, 1)
    #print_matrix_row(hGradQuery, seq_len, emb_dim, 1)

    print_matrix(hQuery, seq_len, emb_dim, 1)
    print_matrix_row(hGradKey, seq_len, emb_dim, 1)

    print_sparse_matrix(hAttnScore, hOffsets, hColumns, seq_len, emb_dim, nnz, 1)
    print_matrix(hGradOutput, seq_len, emb_dim, 1)
    print_matrix_row(hGradValue, seq_len, emb_dim, 1)


    



    
    
