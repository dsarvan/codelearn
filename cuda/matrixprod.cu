/* File: matrixprod.cu
 * Name: D.Saravanan
 * Date: 30/03/2023
 * Program computes matrix multiplication on the GPU using CUDA
*/

#include <stdlib.h>

__global__ void matprod(float *a, float *b, float *c, int N) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N && j < N) {
	float res = 0;
	for (size_t n; n < N; n++)
	    res += a[i * N + n] * b[n * N + j];

	c[i * N + j] = res;
    }
}

void initialize(float *m, int N) {
    for (size_t i = 0; i < N; i++)
	m[i] = rand() % 100;
}

int main() {

    /* square matrix dimension (2**10 x 2**10) */
    int N = 1 << 10;
    size_t bytes = N * N * sizeof(bytes);

    /* allocate matrix in unified memory */
    float *a, *b, *c;
    cudaMallocManaged((void **) &a, bytes);
    cudaMallocManaged((void **) &b, bytes);
    cudaMallocManaged((void **) &c, bytes);

    /* initialize input matrix */
    initialize(a, N);
    initialize(b, N);

    int threads = 16;	/* threads per block */
    int blocks = (N + threads - 1) / threads;	/* blocks per grid */

    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);

    /* launch kernel */
    matprod <<< BLOCKS, THREADS >>> (a, b, c, N);

    /* force host to wait on the completion of the kernel */
    cudaDeviceSynchronize();

    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
