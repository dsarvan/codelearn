/* File: matrixadd.cu
 * Name: D.Saravanan
 * Date: 02/04/2023
 * Program computes matrix addition on the GPU using CUDA 
*/

#include <stdlib.h>

__global__ void matadd(float *a, float *b, float *c, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < N && j < N)
        c[i][j] = a[i][j] + b[i][j];
}

void initialize(float *m, int N) {
    for (size_t i = 0; i < N; i++)
        m[i] = rand() % 100;
}

int main() {

    int N = 1 << 10;
    size_t bytes = N * N * sizeof(bytes);

    float *a, *b, *c;
    cudaMallocManaged((void **) &a, bytes);
    cudaMallocManaged((void **) &b, bytes);
    cudaMallocManaged((void **) &c, bytes);

    initialize(a, N);
    initialize(b, N);

    int threads = 16;
    int blocks = (N + threads - 1) / threads;

    dim3 Threads(threads, threads);
    dim3 Blocks(blocks, blocks);

    matadd <<< Blocks, Threads >>> (a, b, c, N);

    cudaDeviceSynchronize();

    /* free device memory for a, b, c */
    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
