/* File: vectoradd.cu
 * Name: D.Saravanan
 * Date: 30/03/2023
 * Program computes the sum of two arrays on the GPU using CUDA
*/

#include <stdlib.h>

__global__ void vecadd(float *a, float *b, float *c, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
	c[i] = a[i] + b[i];
}

void initialize(float *a, int N) {
    for (size_t i = 0; i < N; i++)
	a[i] = rand() % 100;
}

int main() {

    /* vector dimension (2**20) */
    int N = 1 << 20;
    size_t bytes = N * sizeof(bytes);

    /* allocate vector in unified memory */
    float *a, *b, *c;
    cudaMallocManaged((void **) &a, bytes);
    cudaMallocManaged((void **) &b, bytes);
    cudaMallocManaged((void **) &c, bytes);

    /* initialize input vectors */
    initialize(a, N);
    initialize(b, N);

    int threads = 256;	 /* threads per block */
    int blocks = (N + threads - 1) / threads;	/* blocks per grid */

    /* launch kernel */
    vecadd <<< blocks, threads >>> (a, b, c, N);

    /* force host to wait on the completion of the kernel */
    cudaDeviceSynchronize();

    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
