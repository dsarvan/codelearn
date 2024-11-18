/* File: 1dgrid1dblock.cu
 * Name: D.Saravanan
 * Date: 18/11/2024
 * Program compute thread index with 1D grid of 1D blocks
 *
 * $ nvcc -o 1dgrid1dblock 1dgrid1dblock.cu
 * $ ./1dgrid1dblock
*/

#include <stdio.h>

/* 1D grid of 1D blocks */
__global__ void threadId_1D_1D() {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("threadId: %d = blockIdx.x: %d * blockDim.x: %d + threadIdx.x: %d\n",\
            threadId, blockIdx.x, blockDim.x, threadIdx.x);
}

int main() {
    cudaSetDevice(0);

    dim3 gridDim, blockDim;

    gridDim.x = 2;
    blockDim.x = 4;

    threadId_1D_1D<<<gridDim, blockDim>>>();

    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}
