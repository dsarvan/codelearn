/* File: 3dgrid2dblock.cu
 * Name: D.Saravanan
 * Date: 25/11/2024
 * Program compute thread index with 3D grid of 2D blocks
 *
 * $ nvcc -o 3dgrid2dblock 3dgrid2dblock.cu
 * $ ./3dgrid2dblock
*/

#include <stdio.h>

/* 3D grid of 2D blocks */
__global__ void threadId_3D_2D() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x \
                  + gridDim.x * gridDim.y * blockIdx.z;

    int threadId = blockId * (blockDim.x * blockDim.y) \
                   + (threadIdx.y * blockDim.x) + threadIdx.x;

    printf("blockId: %d = blockIdx.x: %d + blockIdx.y: %d * gridDim.x: %d "
           "+ gridDim.x: %d * gridDim.y: %d * blockIdx.x: %d\n", blockId,\
           blockIdx.x, blockIdx.y, gridDim.x, gridDim.x, gridDim.y, blockIdx.z);
    printf("\n");
    printf("threadId: %d = blockId: %d + (blockDim.x: %d * blockDim.y: %d) "
           "+ (threadIdx.y: %d * blockDim.x: %d) + threadIdx.x: %d\n", threadId,\
           blockId, blockDim.x, blockDim.y, threadIdx.y, blockDim.x, threadIdx.x);
}

int main() {
    cudaSetDevice(0);

    dim3 gridDim, blockDim;

    gridDim.x = 2;
    gridDim.y = 3;
    gridDim.z = 2;

    blockDim.x = 4;
    blockDim.y = 2;

    threadId_3D_2D<<<gridDim, blockDim>>>();

    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}
