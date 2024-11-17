/* File: threadindex.cu
 * Name: D.Saravanan
 * Date: 25/09/2024
 * Program compute global thread index using the block and thread indices
 *
 * $ nvcc -o threadindex threadindex.cu
 * $ ./threadindex
 *
 * threadIdx.x: 0, blockIdx.x: 0, blockDim.x: 2, threadId: 0
 * threadIdx.x: 1, blockIdx.x: 0, blockDim.x: 2, threadId: 1
 * threadIdx.x: 0, blockIdx.x: 1, blockDim.x: 2, threadId: 2
 * threadIdx.x: 1, blockIdx.x: 1, blockDim.x: 2, threadId: 3
*/

#include <stdio.h>

__global__ void threadIndex() {
  /* threadIdx.x: the thread id with respect to the thread's block
                  [from 0 to (thread count per block - 1)]

     blockIdx.x: the block id with respect to the grid (all blocks in the kernel)
                 [from 0 to (number of blocks launched - 1)]

     blockDim.x: the number of threads in a block (block's dimension) [single value]

     threadId: the id of the thread with respect to the whole kernel
               [from 0 to (kernel thread count - 1)]
  */

  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  printf("threadIdx.x: %d, blockIdx.x: %d, blockDim.x: %d, threadId: %d\n", threadIdx.x, blockIdx.x, blockDim.x, threadId);
}


int main(int argc, char *argv[]) {

  /* set which device should be used */
  /* the code will default to 0 if not called though */
  cudaSetDevice(0);

  /* call a device function from the host: kernel launch */
  threadIndex<<<2, 2>>>();

  /* this call waits for all of the submitted GPU work to complete */
  cudaDeviceSynchronize();

  /* destorys and cleans up all resources associated with the current device */
  /* It will reset the device immediately. It is the user's responsibility to ensure that the device work has completed. */
  cudaDeviceReset();

  return 0;
}
