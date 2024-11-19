/* File: 1dgrid2dblock.cu
 * Name: D.Saravanan
 * Date: 19/11/2024
 * Program compute thread index with 1D grid of 2D blocks
 *
 * $ nvcc -o 1dgrid2dblock 1dgrid2dblock.cu
 * $ ./1dgrid2dblock
*/

#include <stdio.h>

/* 1D grid of 2D blocks */
__global__ void threadId_1D_2D() {
    int threadId = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y \
                   * blockDim.x + threadIdx.x;

    printf("threadId: %d = blockIdx.x: %d * blockDim.x: %d * blockDim.y: %d "
           "+ threadIdx.y: %d * blockDim.x: %d + threadIdx.x: %d\n", threadId,\
           blockIdx.x, blockDim.x, blockDim.y, threadIdx.y, blockDim.x, threadIdx.x);
}

int main() {
    cudaSetDevice(0);

    dim3 gridDim, blockDim;

    gridDim.x = 2;

    blockDim.x = 4;
    blockDim.y = 4;

    threadId_1D_2D<<<gridDim, blockDim>>>();

    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}
