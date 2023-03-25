/* File: vectorinc.cu
 * Name: D.Saravanan
 * Date: 25/03/2023
 * Program to parallelize for loops 
*/

#include <stdio.h>

__global__ void vecinc(int *a, int N) {
    int i = threadIdx.x;

    if (i < N)
	    a[i] = a[i] + 1;
}

int main() {

    int N = 1000000000;

    /* allocate input vector h_a in host memory */
    int *h_a = (int *) malloc(N * sizeof(int));

    /* allocate vector in device memory */
    int *d_a;
    cudaMalloc((void **) &d_a, N * sizeof(int));

    /* copy vector from host memory to device memory */
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid_size(1);
    dim3 block_size(N);

    /* launch kernel */
    vecinc <<< grid_size, block_size >>> (d_a, N);

    /* force host to wait on the completion of the kernel */
    cudaDeviceSynchronize();

    /* copy vector from device memory to host memory */
    cudaMemcpy(h_a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    free(h_a);

    return 0;
}
