/* File: threadidx.cu
 * Name: D.Saravanan
 * Date: 24/09/2024
 * Program to print thread index per block of 1D grid
 *
 * $ nvcc -o threadidx threadidx.cu
 * $ ./threadidx 5
 *
 * threadidx: 0
 * threadidx: 1
 * threadidx: 2
 * threadidx: 3
 * threadidx: 4
*/

#include <stdio.h>
#include <cuda.h>


__global__ void threadidx() {
    printf("threadidx: %d\n", threadIdx.x);
}


int main(int argc, char *argv[]) {
    int thread = strtol(argv[1], NULL, 10);
    threadidx<<<1, thread>>>();
    cudaDeviceSynchronize();
    return 0;
}
