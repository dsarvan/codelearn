/* File: 2dgrid1dblock.cu
 * Name: D.Saravanan
 * Date: 21/11/2024
 * Program compute thread index with 2D grid of 1D blocks
 *
 * $ nvcc -o 2dgrid1dblock 2dgrid1dblock.cu
 * $ ./2dgrid1dblock
*/

#include <stdio.h>

/* 2D grid of 1D blocks */
__global__ void threadId_2D_1D() {
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;

    printf("blockId: %d = blockIdx.y: %d * gridDim.x: %d + blockIdx.x: %d\n",\
            blockId, blockIdx.y, gridDim.x, blockIdx.x);
    printf("\n");
    printf("threadId: %d = blockId: %d * blockDim.x: %d + threadIdx.x: %d\n",\
            threadId, blockId, blockDim.x, threadIdx.x);
}

int main() {
    cudaSetDevice(0);

    dim3 gridDim, blockDim;

    gridDim.x = 2;
    gridDim.y = 3;

    blockDim.x = 4;

    threadId_2D_1D<<<gridDim, blockDim>>>();

    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}
