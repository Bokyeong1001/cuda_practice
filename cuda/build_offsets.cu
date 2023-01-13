#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void build_sp_offsets_kernel(int *dOffsets, int block_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    dOffsets[i] = i * block_size;
}

void build_sp_offsets_function(int *dOffsets, int seq_len, int block_size){
    build_sp_offsets_kernel<<<1, seq_len+1>>>(dOffsets, block_size);
}

__global__ void build_sp_columns_kernel(int *dColumns, int block_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t_id = threadIdx.x;
    int column;
    int idx = i * block_size * block_size;
    for(int m = 0; m < block_size; m++){
        column = block_size*t_id;
        for(int n = 0; n < block_size; n++){
            dColumns[idx+(m*block_size)+n] = column;
            column += 1;
        }
    }
}

void build_sp_columns_function(int *dColumns, int seq_len, int block_size, int num_batches){
    build_sp_columns_kernel<<<num_batches, seq_len/block_size>>>(dColumns, block_size);
}

int main(){
    int *hOffsets, *hColumns, *dOffsets, *dColumns;
    int seq_len = 4;
    int block_size = 2;
    int num_batches = 2;
    int nnz = seq_len * block_size;
    cudaMallocHost((void **) &hOffsets, (seq_len + 1) * sizeof(int));
    cudaMallocHost((void **) &hColumns, nnz * num_batches * sizeof(int));
    cudaMalloc((void**) &dOffsets, (seq_len + 1) * sizeof(int));
    cudaMalloc((void**) &dColumns, nnz * num_batches * sizeof(int));
    build_sp_offsets_function(dOffsets, seq_len, block_size);
    build_sp_columns_function(dColumns, seq_len, block_size, num_batches);
    cudaMemcpy(hOffsets, dOffsets, (seq_len + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hColumns, dColumns, nnz * num_batches * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i< num_batches*seq_len*block_size; i++){
        printf("%d\n", hColumns[i]);
    }
}