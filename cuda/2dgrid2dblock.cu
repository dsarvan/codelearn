/* File: 2dgrid2dblock.cu
 * Name: D.Saravanan
 * Date: 22/11/2024
 * Program compute thread index with 2D grid of 2D blocks
 *
 * $ nvcc -o 2dgrid2dblock 2dgrid2dblock.cu
 * $ ./2dgrid2dblock
*/

#include <stdio.h>

/* 2D grid of 2D blocks */
__global__ void threadId_2D_2D() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) \
                   + (threadIdx.y * blockDim.x) + threadIdx.x;

    printf("blockId: %d = blockIdx.x: %d + blockIdx.y: %d * gridDim.x: %d\n",\
            blockId, blockIdx.x, blockIdx.y, gridDim.x);
    printf("\n");
    printf("threadId: %d = blockId: %d * (blockDim.x: %d * blockDim.y: %d) "
           "+ (threadIdx.y: %d * blockDim.x: %d) + threadIdx.x: %d\n", threadId,\
           blockId, blockDim.x, blockDim.y, threadIdx.y, blockDim.x, threadIdx.x);
}

int main() {
    cudaSetDevice(0);

    dim3 gridDim, blockDim;

    gridDim.x = 2;
    gridDim.y = 3;

    blockDim.x = 4;
    blockDim.y = 2;

    threadId_2D_2D<<<gridDim, blockDim>>>();

    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}
