import torch
from torch import nn
import ctypes
from ctypes import *
import math
import nvidia_dlprof_pytorch_nvtx
nvidia_dlprof_pytorch_nvtx.init()

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
    int seq_len, int emb_dim, int num_batches)"""

def get_attn_backward():
    dll = ctypes.CDLL('./attn_backward.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.attn_backward
    func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    c_int, c_int, c_int]
    return func

__cuda_attn_backward = get_attn_backward()

if __name__ == '__main__':
    seq_len = 4
    emb_dim = 4
    num_batches = 1
    
    input_size = (num_batches, seq_len, emb_dim)
    attn_size = (num_batches, seq_len, seq_len)

    hQuery = torch.ones(input_size, dtype=torch.float32, device='cuda')
    hKey = torch.ones(input_size, dtype=torch.float32, device='cuda')
    hValue = torch.rand(input_size, dtype=torch.float32, device='cuda')
    hAttnScore = torch.rand(attn_size, dtype=torch.float32, device='cuda')
    hGradOutput = torch.rand(input_size, dtype=torch.float32, device='cuda')
    hGradAttnScore = torch.zeros(attn_size, dtype=torch.float32, device='cuda')
    hGradAttnScoreScale = torch.zeros(attn_size, dtype=torch.float32, device='cuda')
    hGradSum = torch.zeros((num_batches, seq_len), dtype=torch.float32, device='cuda')
    hGradAttnScale = torch.zeros(attn_size, dtype=torch.float32, device='cuda')
    hGradAttn = torch.rand(attn_size, dtype=torch.float32, device='cuda')
    hGradQuery = torch.zeros(input_size, dtype=torch.float32, device='cuda')
    hGradKey = torch.zeros(input_size, dtype=torch.float32, device='cuda')
    hGradValue = torch.zeros(input_size, dtype=torch.float32, device='cuda')
    tmp_hGradOutput = hGradOutput.permute(0,2,1)

    hQuery_p = hQuery.contiguous().data_ptr()
    hKey_p = hKey.contiguous().data_ptr()
    hValue_p = hValue.contiguous().data_ptr()
    hAttnScore_p = hAttnScore.contiguous().data_ptr()
    hGradOutput_p = tmp_hGradOutput.contiguous().data_ptr()
    hGradAttnScore_p = hGradAttnScore.contiguous().data_ptr()
    hGradAttnScoreScale_p = hGradAttnScoreScale.contiguous().data_ptr()
    hGradSum_p = hGradSum.contiguous().data_ptr()
    hGradAttnScale_p = hGradAttnScale.contiguous().data_ptr()
    hGradAttn_p = hGradAttn.contiguous().data_ptr()
    hGradQuery_p = hGradQuery.contiguous().data_ptr()
    hGradKey_p = hGradKey.contiguous().data_ptr()
    hGradValue_p = hGradValue.contiguous().data_ptr()

    """(float *hQuery, float *hKey, float *hValue, float *hAttnScore, float *hGradOutput, float *hGradAttnScore, float *hGradAttnScoreScale, 
                    float *hGradSum, float *hGradAttnScale, float *hGradAttn, float *hGradQuery, float *hGradKey, float *hGradValue,
                    int seq_len, int emb_dim, int num_batches)"""

    __cuda_attn_backward(hQuery_p, hKey_p, hValue_p, hAttnScore_p, hGradOutput_p, hGradAttnScore_p, hGradAttnScoreScale_p, 
                        hGradSum_p, hGradAttnScale_p, hGradAttn_p, hGradQuery_p, hGradKey_p, hGradValue_p,
                        seq_len, emb_dim, num_batches)
    
    #print(hGradOutput)
    #print(hValue)
    #print("Grad Attn Score = ")
    #print(hGradAttnScore)
    #print(hGradAttnScore)
    #print(hAttnScore)
    #print(hGradOutput)
    #print(hGradAttnScoreScale)
    #print(hGradSum)
    print(hGradAttn)
    #print(hKey)
    print(hGradQuery)
    #print(hQuery)
    print(hGradKey)
    print(hGradValue)

    grad_attn_score = torch.matmul(hGradOutput, hValue.permute(0,2,1))
    print("Torch Grad Attn Score = ")
    #print(grad_attn_score)
    #print(grad_attn_score)
    #print(hAttnScore)
    grad_attn_score_scale = grad_attn_score * hAttnScore
    #print(grad_attn_score_scale)
    sum_gradient = torch.sum(grad_attn_score_scale, dim=-1, keepdim=True)
    #print(sum_gradient)
    grad_attn_scale = grad_attn_score_scale - (hAttnScore * sum_gradient)
    grad_attn = grad_attn_scale / math.sqrt(emb_dim)
    print(hGradAttn)
    grad_qeury = torch.matmul(hGradAttn, hKey)
    print(grad_qeury)
    grad_key = torch.matmul(hGradAttn.permute(0,2,1), hQuery)
    print(grad_key)
    tmp_attn_score = hAttnScore.permute(0,2,1)
    grad_value = torch.matmul(tmp_attn_score, hGradOutput)
    print(grad_value)



    
    
