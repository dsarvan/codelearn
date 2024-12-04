#!/usr/bin/env python
# File: 2dgrid1dblock.py
# Name: D.Saravanan
# Date: 04/12/2024
# Script compute thread index with 2D grid of 1D blocks

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

kernel = """
#include <stdio.h>

/* 2D grid of 1D blocks */
__global__ void threadId_2D_1D() {
	int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	int threadId = blockId * blockDim.x + threadIdx.x;

	printf("blockId: %d = blockIdx.y: %d * gridDim.x: %d + blockIdx.x: %d\\n",\
			blockId, blockIdx.y, gridDim.x, blockIdx.x);
	printf("\\n");
	printf("threadId: %d = blockId: %d * blockDim.x: %d + threadIdx.x: %d\\n",\
			threadId, blockId, blockDim.x, threadIdx.x);
}
"""

mod = SourceModule(kernel)
threadId_2D_1D = mod.get_function("threadId_2D_1D")
threadId_2D_1D(grid=(2,3,1), block=(4,1,1))

"""
Output:

blockId: 2 = blockIdx.y: 1 * gridDim.x: 2 + blockIdx.x: 0
blockId: 2 = blockIdx.y: 1 * gridDim.x: 2 + blockIdx.x: 0
blockId: 2 = blockIdx.y: 1 * gridDim.x: 2 + blockIdx.x: 0
blockId: 2 = blockIdx.y: 1 * gridDim.x: 2 + blockIdx.x: 0
blockId: 0 = blockIdx.y: 0 * gridDim.x: 2 + blockIdx.x: 0
blockId: 0 = blockIdx.y: 0 * gridDim.x: 2 + blockIdx.x: 0
blockId: 0 = blockIdx.y: 0 * gridDim.x: 2 + blockIdx.x: 0
blockId: 0 = blockIdx.y: 0 * gridDim.x: 2 + blockIdx.x: 0
blockId: 5 = blockIdx.y: 2 * gridDim.x: 2 + blockIdx.x: 1
blockId: 5 = blockIdx.y: 2 * gridDim.x: 2 + blockIdx.x: 1
blockId: 5 = blockIdx.y: 2 * gridDim.x: 2 + blockIdx.x: 1
blockId: 5 = blockIdx.y: 2 * gridDim.x: 2 + blockIdx.x: 1
blockId: 1 = blockIdx.y: 0 * gridDim.x: 2 + blockIdx.x: 1
blockId: 1 = blockIdx.y: 0 * gridDim.x: 2 + blockIdx.x: 1
blockId: 1 = blockIdx.y: 0 * gridDim.x: 2 + blockIdx.x: 1
blockId: 1 = blockIdx.y: 0 * gridDim.x: 2 + blockIdx.x: 1
blockId: 3 = blockIdx.y: 1 * gridDim.x: 2 + blockIdx.x: 1
blockId: 3 = blockIdx.y: 1 * gridDim.x: 2 + blockIdx.x: 1
blockId: 3 = blockIdx.y: 1 * gridDim.x: 2 + blockIdx.x: 1
blockId: 3 = blockIdx.y: 1 * gridDim.x: 2 + blockIdx.x: 1
blockId: 4 = blockIdx.y: 2 * gridDim.x: 2 + blockIdx.x: 0
blockId: 4 = blockIdx.y: 2 * gridDim.x: 2 + blockIdx.x: 0
blockId: 4 = blockIdx.y: 2 * gridDim.x: 2 + blockIdx.x: 0
blockId: 4 = blockIdx.y: 2 * gridDim.x: 2 + blockIdx.x: 0


threadId: 8 = blockId: 2 * blockDim.x: 4 + threadIdx.x: 0
threadId: 9 = blockId: 2 * blockDim.x: 4 + threadIdx.x: 1
threadId: 10 = blockId: 2 * blockDim.x: 4 + threadIdx.x: 2
threadId: 11 = blockId: 2 * blockDim.x: 4 + threadIdx.x: 3
threadId: 20 = blockId: 5 * blockDim.x: 4 + threadIdx.x: 0
threadId: 21 = blockId: 5 * blockDim.x: 4 + threadIdx.x: 1
threadId: 22 = blockId: 5 * blockDim.x: 4 + threadIdx.x: 2
threadId: 23 = blockId: 5 * blockDim.x: 4 + threadIdx.x: 3
threadId: 0 = blockId: 0 * blockDim.x: 4 + threadIdx.x: 0
threadId: 1 = blockId: 0 * blockDim.x: 4 + threadIdx.x: 1
threadId: 2 = blockId: 0 * blockDim.x: 4 + threadIdx.x: 2
threadId: 3 = blockId: 0 * blockDim.x: 4 + threadIdx.x: 3
threadId: 4 = blockId: 1 * blockDim.x: 4 + threadIdx.x: 0
threadId: 5 = blockId: 1 * blockDim.x: 4 + threadIdx.x: 1
threadId: 6 = blockId: 1 * blockDim.x: 4 + threadIdx.x: 2
threadId: 7 = blockId: 1 * blockDim.x: 4 + threadIdx.x: 3
threadId: 12 = blockId: 3 * blockDim.x: 4 + threadIdx.x: 0
threadId: 13 = blockId: 3 * blockDim.x: 4 + threadIdx.x: 1
threadId: 14 = blockId: 3 * blockDim.x: 4 + threadIdx.x: 2
threadId: 15 = blockId: 3 * blockDim.x: 4 + threadIdx.x: 3
threadId: 16 = blockId: 4 * blockDim.x: 4 + threadIdx.x: 0
threadId: 17 = blockId: 4 * blockDim.x: 4 + threadIdx.x: 1
threadId: 18 = blockId: 4 * blockDim.x: 4 + threadIdx.x: 2
threadId: 19 = blockId: 4 * blockDim.x: 4 + threadIdx.x: 3

"""
