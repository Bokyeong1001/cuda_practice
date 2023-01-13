import torch
from torch import nn
import ctypes
from ctypes import *
import math
import nvidia_dlprof_pytorch_nvtx
nvidia_dlprof_pytorch_nvtx.init()

def get_attn_forward():
    dll = ctypes.CDLL('./attn_forward.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.attn_forward
    func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_int, c_int, c_int]
    return func

__cuda_attn_forward = get_attn_forward()

if __name__ == '__main__':
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    seq_len = 4
    emb_dim = 4
    num_batches = 1

    input_size = (num_batches, seq_len, emb_dim)
    attn_size = (num_batches, seq_len, seq_len)

    hQuery = torch.zeros(input_size, dtype=torch.float32, device='cuda')
    hKey = torch.zeros(input_size, dtype=torch.float32, device='cuda')
    hValue = torch.zeros(input_size, dtype=torch.float32, device='cuda')
    hAttn = torch.zeros(attn_size, dtype=torch.float32, device='cuda')
    hAttnOut = torch.zeros(attn_size, dtype=torch.float32, device='cuda')
    hOut = torch.zeros(input_size, dtype=torch.float32, device='cuda')

    for b in range(num_batches):
        for j in range(seq_len):
            for i in range(emb_dim):
                index = i*seq_len + j + b*seq_len*emb_dim
                hQuery[b][j][i] = i%4
                hKey[b][j][i] = i%4 + 1
                hValue[b][j][i] = i%4

    tmp_hQuery = hQuery.permute(0,2,1)
    tmp_hValue = hValue.permute(0,2,1)
    torch_key = hKey.permute(0,2,1)
    
    print("Query = ")
    print(hQuery)
    print("Key = ")
    print(hKey)
    print("Value = ")
    print(hValue)

    hQuery_p = tmp_hQuery.contiguous().data_ptr()
    hKey_p = hKey.contiguous().data_ptr()
    hValue_p = tmp_hValue.contiguous().data_ptr()
    hAttn_p = hAttn.contiguous().data_ptr()
    hAttnOut_p = hAttnOut.contiguous().data_ptr()
    hOut_p = hOut.contiguous().data_ptr()
    #with torch.autograd.profiler.emit_nvtx():
    __cuda_attn_forward(hQuery_p, hKey_p, hValue_p, hAttn_p, hAttnOut_p, hOut_p, seq_len, emb_dim, num_batches)
    
    print("Attn = ")
    print(hAttn)

    torch_attn = torch.matmul(hQuery, torch_key) 
    print("Torch Attn = ")
    print(torch_attn)

    print("Attn Out = ")
    print(hAttnOut)

    print("Torch Attn Out = ")
    softmax = torch.nn.Softmax(dim=-1)
    torch_attn_out = softmax(torch_attn/math.sqrt(emb_dim))
    print(torch_attn_out)
    
    print("\nOut = ")
    print(hOut.permute(0,2,1))
    torch_out = torch.matmul(torch_attn_out, hValue)
    print("Torch Out = ")
    print(torch_out)
