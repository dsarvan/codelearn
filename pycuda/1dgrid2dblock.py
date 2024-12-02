#!/usr/bin/env python
# File: 1dgrid2dblock.py
# Name: D.Saravanan
# Date: 02/12/2024
# Script compute thread index with 1D grid of 2D blocks

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

kernel = """
#include <stdio.h>

/* 1D grid of 2D blocks */
__global__ void threadId_1D_2D() {
	int threadId = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y \
	               * blockDim.x + threadIdx.x;

	printf("threadId: %d = blockIdx.x: %d * blockDim.x: %d * blockDim.y: %d "
	       "+ threadIdx.y: %d * blockDim.x: %d + threadIdx.x: %d\\n", threadId,\
		   blockIdx.x, blockDim.x, blockDim.y, threadIdx.y, blockDim.x, threadIdx.x);
}
"""

mod = SourceModule(kernel)
threadId_1D_2D = mod.get_function("threadId_1D_2D")
threadId_1D_2D(grid=(2,1,1), block=(4,4,1))

"""
Output:

threadId: 0 = blockIdx.x: 0 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 0 * blockDim.x: 4 + threadIdx.x: 0
threadId: 1 = blockIdx.x: 0 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 0 * blockDim.x: 4 + threadIdx.x: 1
threadId: 2 = blockIdx.x: 0 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 0 * blockDim.x: 4 + threadIdx.x: 2
threadId: 3 = blockIdx.x: 0 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 0 * blockDim.x: 4 + threadIdx.x: 3
threadId: 4 = blockIdx.x: 0 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 1 * blockDim.x: 4 + threadIdx.x: 0
threadId: 5 = blockIdx.x: 0 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 1 * blockDim.x: 4 + threadIdx.x: 1
threadId: 6 = blockIdx.x: 0 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 1 * blockDim.x: 4 + threadIdx.x: 2
threadId: 7 = blockIdx.x: 0 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 1 * blockDim.x: 4 + threadIdx.x: 3
threadId: 8 = blockIdx.x: 0 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 2 * blockDim.x: 4 + threadIdx.x: 0
threadId: 9 = blockIdx.x: 0 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 2 * blockDim.x: 4 + threadIdx.x: 1
threadId: 10 = blockIdx.x: 0 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 2 * blockDim.x: 4 + threadIdx.x: 2
threadId: 11 = blockIdx.x: 0 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 2 * blockDim.x: 4 + threadIdx.x: 3
threadId: 12 = blockIdx.x: 0 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 3 * blockDim.x: 4 + threadIdx.x: 0
threadId: 13 = blockIdx.x: 0 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 3 * blockDim.x: 4 + threadIdx.x: 1
threadId: 14 = blockIdx.x: 0 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 3 * blockDim.x: 4 + threadIdx.x: 2
threadId: 15 = blockIdx.x: 0 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 3 * blockDim.x: 4 + threadIdx.x: 3
threadId: 16 = blockIdx.x: 1 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 0 * blockDim.x: 4 + threadIdx.x: 0
threadId: 17 = blockIdx.x: 1 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 0 * blockDim.x: 4 + threadIdx.x: 1
threadId: 18 = blockIdx.x: 1 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 0 * blockDim.x: 4 + threadIdx.x: 2
threadId: 19 = blockIdx.x: 1 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 0 * blockDim.x: 4 + threadIdx.x: 3
threadId: 20 = blockIdx.x: 1 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 1 * blockDim.x: 4 + threadIdx.x: 0
threadId: 21 = blockIdx.x: 1 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 1 * blockDim.x: 4 + threadIdx.x: 1
threadId: 22 = blockIdx.x: 1 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 1 * blockDim.x: 4 + threadIdx.x: 2
threadId: 23 = blockIdx.x: 1 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 1 * blockDim.x: 4 + threadIdx.x: 3
threadId: 24 = blockIdx.x: 1 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 2 * blockDim.x: 4 + threadIdx.x: 0
threadId: 25 = blockIdx.x: 1 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 2 * blockDim.x: 4 + threadIdx.x: 1
threadId: 26 = blockIdx.x: 1 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 2 * blockDim.x: 4 + threadIdx.x: 2
threadId: 27 = blockIdx.x: 1 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 2 * blockDim.x: 4 + threadIdx.x: 3
threadId: 28 = blockIdx.x: 1 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 3 * blockDim.x: 4 + threadIdx.x: 0
threadId: 29 = blockIdx.x: 1 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 3 * blockDim.x: 4 + threadIdx.x: 1
threadId: 30 = blockIdx.x: 1 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 3 * blockDim.x: 4 + threadIdx.x: 2
threadId: 31 = blockIdx.x: 1 * blockDim.x: 4 * blockDim.y: 4 + threadIdx.y: 3 * blockDim.x: 4 + threadIdx.x: 3

"""
