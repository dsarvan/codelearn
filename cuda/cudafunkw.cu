/* File: cudafunkw.cu
 * Name: D.Saravanan
 * Date: 26/09/2024
 * CUDA function keywords program
*/

#include <stdio.h>

/* __device__ keyword specifies a function that is
   run on the device and called from a kernel (1a) */
__device__ void gpuKernelFunction() {
    printf("Printing from the GPU!\n");
}

/* kernel that calls a device function (1b) */
__global__ void kernelFunction1() {
    gpuKernelFunction();
}

/* __host__ __device__ keywords can be specified if the function
   needs to be available to both the host and device (2a) */
__host__ __device__ void versatileFunction() {
    printf("Printing from the CPU or the GPU!\n");
}

/* kernel that calls a device function (2b) */
__global__ void kernelFunction2() {
    versatileFunction();
}


int main() {
    cudaSetDevice(0);

    /* kernel launch, that will print from a function called by device (1b -> 1a) */
    printf("Launching kernel (1b):\n");
    kernelFunction1<<<1, 1>>>();

    /* call function from the host (2a) */
    printf("Calling host function (2a):\n");
    versatileFunction();

    /* call the same function from the device (2b -> 2a) */
    printf("Launching kernel (2b):\n");
    kernelFunction2<<<1, 1>>>();

    cudaDeviceSynchronize();

    return 0;
}
