#!/usr/bin/env python
# File: 3dgrid1dblock.py
# Name: D.Saravanan
# Date: 07/12/2024
# Script compute thread index with 3D grid of 1D blocks

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

kernel = """
#include <stdio.h>

/* 3D grid of 1D blocks */
__global__ void threadId_3D_1D() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x \
				  + gridDim.x * gridDim.y * blockIdx.z;

	int threadId = blockId * blockDim.x + threadIdx.x;

	printf("blockId: %d = blockIdx.x: %d + blockIdx.y: %d * gridDim.x: %d "
		   "+ gridDim.x: %d * gridDim.y: %d * blockIdx.z: %d\\n", blockId,\
		   blockIdx.x, blockIdx.y, gridDim.x, gridDim.x, gridDim.y, blockIdx.z);
	printf("\\n");
	printf("threadId: %d = blockId: %d * blockDim.x: %d + threadIdx.x: %d\\n",\
			threadId, blockId, blockDim.x, threadIdx.x);
}
"""

mod = SourceModule(kernel)
threadId_3D_1D = mod.get_function("threadId_3D_1D")
threadId_3D_1D(grid=(2,3,2), block=(4,1,1))

"""
Output:

blockId: 2 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 2 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 2 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 2 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 7 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 7 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 7 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 7 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 10 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 10 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 10 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 10 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 0 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 0 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 0 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 0 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 5 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 5 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 5 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 5 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 11 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 11 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 11 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 11 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 6 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 6 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 6 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 6 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 9 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 9 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 9 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 9 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 0
blockId: 8 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 8 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 8 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1
blockId: 8 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2 + gridDim.x: 2 * gridDim.y: 3 * blockIdx.z: 1


threadId: 28 = blockId: 7 * blockDim.x: 4 + threadIdx.x: 0
threadId: 29 = blockId: 7 * blockDim.x: 4 + threadIdx.x: 1
threadId: 30 = blockId: 7 * blockDim.x: 4 + threadIdx.x: 2
threadId: 31 = blockId: 7 * blockDim.x: 4 + threadIdx.x: 3
threadId: 8 = blockId: 2 * blockDim.x: 4 + threadIdx.x: 0
threadId: 9 = blockId: 2 * blockDim.x: 4 + threadIdx.x: 1
threadId: 10 = blockId: 2 * blockDim.x: 4 + threadIdx.x: 2
threadId: 11 = blockId: 2 * blockDim.x: 4 + threadIdx.x: 3
threadId: 44 = blockId: 11 * blockDim.x: 4 + threadIdx.x: 0
threadId: 45 = blockId: 11 * blockDim.x: 4 + threadIdx.x: 1
threadId: 46 = blockId: 11 * blockDim.x: 4 + threadIdx.x: 2
threadId: 47 = blockId: 11 * blockDim.x: 4 + threadIdx.x: 3
threadId: 40 = blockId: 10 * blockDim.x: 4 + threadIdx.x: 0
threadId: 41 = blockId: 10 * blockDim.x: 4 + threadIdx.x: 1
threadId: 42 = blockId: 10 * blockDim.x: 4 + threadIdx.x: 2
threadId: 43 = blockId: 10 * blockDim.x: 4 + threadIdx.x: 3
threadId: 24 = blockId: 6 * blockDim.x: 4 + threadIdx.x: 0
threadId: 25 = blockId: 6 * blockDim.x: 4 + threadIdx.x: 1
threadId: 26 = blockId: 6 * blockDim.x: 4 + threadIdx.x: 2
threadId: 27 = blockId: 6 * blockDim.x: 4 + threadIdx.x: 3
threadId: 0 = blockId: 0 * blockDim.x: 4 + threadIdx.x: 0
threadId: 1 = blockId: 0 * blockDim.x: 4 + threadIdx.x: 1
threadId: 2 = blockId: 0 * blockDim.x: 4 + threadIdx.x: 2
threadId: 3 = blockId: 0 * blockDim.x: 4 + threadIdx.x: 3
threadId: 20 = blockId: 5 * blockDim.x: 4 + threadIdx.x: 0
threadId: 21 = blockId: 5 * blockDim.x: 4 + threadIdx.x: 1
threadId: 22 = blockId: 5 * blockDim.x: 4 + threadIdx.x: 2
threadId: 23 = blockId: 5 * blockDim.x: 4 + threadIdx.x: 3
threadId: 36 = blockId: 9 * blockDim.x: 4 + threadIdx.x: 0
threadId: 37 = blockId: 9 * blockDim.x: 4 + threadIdx.x: 1
threadId: 38 = blockId: 9 * blockDim.x: 4 + threadIdx.x: 2
threadId: 39 = blockId: 9 * blockDim.x: 4 + threadIdx.x: 3
threadId: 32 = blockId: 8 * blockDim.x: 4 + threadIdx.x: 0
threadId: 33 = blockId: 8 * blockDim.x: 4 + threadIdx.x: 1
threadId: 34 = blockId: 8 * blockDim.x: 4 + threadIdx.x: 2
threadId: 35 = blockId: 8 * blockDim.x: 4 + threadIdx.x: 3
threadId: 16 = blockId: 4 * blockDim.x: 4 + threadIdx.x: 0
threadId: 17 = blockId: 4 * blockDim.x: 4 + threadIdx.x: 1
threadId: 18 = blockId: 4 * blockDim.x: 4 + threadIdx.x: 2
threadId: 19 = blockId: 4 * blockDim.x: 4 + threadIdx.x: 3
threadId: 12 = blockId: 3 * blockDim.x: 4 + threadIdx.x: 0
threadId: 13 = blockId: 3 * blockDim.x: 4 + threadIdx.x: 1
threadId: 14 = blockId: 3 * blockDim.x: 4 + threadIdx.x: 2
threadId: 15 = blockId: 3 * blockDim.x: 4 + threadIdx.x: 3
threadId: 4 = blockId: 1 * blockDim.x: 4 + threadIdx.x: 0
threadId: 5 = blockId: 1 * blockDim.x: 4 + threadIdx.x: 1
threadId: 6 = blockId: 1 * blockDim.x: 4 + threadIdx.x: 2
threadId: 7 = blockId: 1 * blockDim.x: 4 + threadIdx.x: 3

"""
