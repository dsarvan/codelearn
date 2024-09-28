/* File: cudathread.cu
 * Name: D.Saravanan
 * Date: 27/09/2024
 * Program to launch functions with more than one thread
*/

#include <stdio.h>

__global__ void kernelFunction1() {
    printf("Printing from kernel function 1.\n");
}

__global__ void kernelFunction2() {
    printf("Printing from kernel function 2.\n");
}

int main() {
    int blocks, threads;

    cudaSetDevice(0);

    printf("------- Kernel Function 1 -------\n");
    blocks = 1;
    threads = 4;

    /* calling a kernel with 1 block that contains 4 threads,
       launching a total of 4 threads */
    kernelFunction1<<<blocks, threads>>>();

    cudaDeviceSynchronize();

    printf("\n");

    printf("------- Kernel Function 2 -------\n");
    blocks = 3;
    threads = 2;

    /* calling a kernel with 3 blocks that each contains 2 threads,
       launching a total of 6 threads */
    kernelFunction2<<<blocks, threads>>>();

    cudaDeviceSynchronize();

    return 0;
}
