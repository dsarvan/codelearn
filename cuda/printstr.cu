/* File: printstr.cu
 * Name: D.Saravanan
 * Date: 25/09/2024
 * Program to print string from the CPU and the GPU
*/

#include <stdio.h>

/* __global__ keyword specifies a device kernel function */
__global__ void kernelFunction() {
    printf("Printing from the GPU!\n");
}

int main() {
    printf("Printing from the CPU!\n);

    /* set which device should be used */
    /* the program will default to 0 if not called though */
    cudaSetDevice(0);

    /* call a device function from the host: kernel launch */
    kernelFunction<<<1, 1>>>();

    /* this call waits for all of the submitted GPU work to complete */
    cudaDeviceSynchronize();

    return 0;
}
