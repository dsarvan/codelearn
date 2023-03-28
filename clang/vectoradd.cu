/* File: vectoradd.cu
 * Name: D.Saravanan
 * Date: 28/03/2023
 * Program computes the sum of two arrays on the GPU using CUDA
*/

#include <math.h>

__global__ void vecadd(float *a, float *b, float *c, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        c[i] = a[i] + b[i];
}

int main() {

    int N = 1000;

    /* allocate input vectors h_a and h_b in host memory */
    float *h_a = (float *) malloc(N * sizeof(float));
    float *h_b = (float *) malloc(N * sizeof(float));
    float *h_c = (float *) malloc(N * sizeof(float));

    /* initialize input vectors */
    for (size_t i = 0; i < N; i++) {
        h_a[i] = sin(i) + cos(i);
        h_b[i] = tan(i);
    }

    /* allocate vector in device memory */
    float *d_a;
    cudaMalloc((void **) &d_a, N * sizeof(float));
    float *d_b;
    cudaMalloc((void **) &d_b, N * sizeof(float));
    float *d_c;
    cudaMalloc((void **) &d_c, N * sizeof(float));

    /* copy vectors from host memory to device memory */
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid_size(1);
    dim3 block_size(N);

    /* launch kernel */
    vecadd <<< grid_size, block_size >>> (d_a, d_b, d_c, N);

    /* force host to wait on the completion of the kernel */
    cudaDeviceSynchronize();

    /* copy vector from device memory to host memory */
    cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    
    return 0;
}
