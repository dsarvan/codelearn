/* File: 1dgrid3dblock.cu
 * Name: D.Saravanan
 * Date: 20/11/2024
 * Program compute thread index with 1D grid of 3D blocks
 *
 * $ nvcc -o 1dgrid3dblock 1dgrid3dblock.cu
 * $ ./1dgrid3dblock
*/

#include <stdio.h>

/* 1D grid of 3D blocks */
__global__ void threadId_1D_3D() {
    int threadId = blockIdx.x * blockDim.x * blockDim.y * blockDim.z + threadIdx.z \
                   * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

    printf("threadId: %d = blockIdx.x: %d * blockDim.x: %d * blockDim.y: %d "
           "* blockDim.z: %d + threadIdx.z: %d * blockDim.y: %d * blockDim.x: %d "
           "+ threadIdx.y: %d * blockDim.x: %d + threadIdx.x: %d\n", threadId,\
           blockIdx.x, blockDim.x, blockDim.y, blockDim.z, threadIdx.z, blockDim.y,\
           blockDim.x, threadIdx.y, blockDim.x, threadIdx.x);
}

int main() {
    cudaSetDevice(0);

    dim3 gridDim, blockDim;

    gridDim.x = 2;

    blockDim.x = 4;
    blockDim.y = 2;
    blockDim.z = 2;

    threadId_1D_3D<<<gridDim, blockDim>>>();

    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}
