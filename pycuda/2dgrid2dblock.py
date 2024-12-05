#!/usr/bin/env python
# File: 2dgrid2dblock.py
# Name: D.Saravanan
# Date: 05/12/2024
# Script compute thread index with 2D grid of 2D blocks

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

kernel = """
#include <stdio.h>

/* 2D grid of 2D blocks */
__global__ void threadId_2D_2D() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) \
				   + (threadIdx.y * blockDim.x) + threadIdx.x;

	printf("blockId: %d = blockIdx.x: %d + blockIdx.y: %d * gridDim.x: %d\\n",\
			blockId, blockIdx.x, blockIdx.y, gridDim.x);
	printf("\\n");
	printf("threadId: %d = blockId: %d * (blockDim.x: %d * blockDim.y: %d) "
		   "+ (threadIdx.y: %d * blockDim.x: %d) + threadIdx.x: %d\\n", threadId,\
		   blockId, blockDim.x, blockDim.y, threadIdx.y, blockDim.x, threadIdx.x);
}
"""

mod = SourceModule(kernel)
threadId_2D_2D = mod.get_function("threadId_2D_2D")
threadId_2D_2D(grid=(2, 3, 1), block=(4, 2, 1))

"""
Output:

blockId: 2 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2
blockId: 2 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2
blockId: 2 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2
blockId: 2 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2
blockId: 2 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2
blockId: 2 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2
blockId: 2 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2
blockId: 2 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 0 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2
blockId: 0 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2
blockId: 0 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2
blockId: 0 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2
blockId: 0 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2
blockId: 0 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2
blockId: 0 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2
blockId: 0 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2
blockId: 5 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2
blockId: 5 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2
blockId: 5 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2
blockId: 5 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2
blockId: 5 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2
blockId: 5 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2
blockId: 5 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2
blockId: 5 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2


threadId: 16 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 0
threadId: 17 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 1
threadId: 18 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 2
threadId: 19 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 3
threadId: 20 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 0
threadId: 21 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 1
threadId: 22 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 2
threadId: 23 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 3
threadId: 40 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 0
threadId: 41 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 1
threadId: 42 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 2
threadId: 43 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 3
threadId: 44 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 0
threadId: 45 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 1
threadId: 46 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 2
threadId: 47 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 3
threadId: 24 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 0
threadId: 25 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 1
threadId: 26 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 2
threadId: 27 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 3
threadId: 28 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 0
threadId: 29 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 1
threadId: 30 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 2
threadId: 31 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 3
threadId: 0 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 0
threadId: 1 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 1
threadId: 2 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 2
threadId: 3 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 3
threadId: 4 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 0
threadId: 5 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 1
threadId: 6 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 2
threadId: 7 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 3
threadId: 8 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 0
threadId: 9 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 1
threadId: 10 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 2
threadId: 11 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 3
threadId: 12 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 0
threadId: 13 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 1
threadId: 14 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 2
threadId: 15 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 3
threadId: 32 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 0
threadId: 33 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 1
threadId: 34 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 2
threadId: 35 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 3
threadId: 36 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 0
threadId: 37 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 1
threadId: 38 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 2
threadId: 39 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 3

"""
