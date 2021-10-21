#include <stdio.h>
#include <stdlib.h>
 __global__ void print_from_gpu(){
     printf("Hello World! from thread {%d,%d}\
        From device\n", blockIdx.x,threadIdx.x);
 }
 int main(){
     printf("Hello World from host!\n");
     print_from_gpu<<<8,128>>>();
     cudaDeviceSynchronize();
     return 0;
 }