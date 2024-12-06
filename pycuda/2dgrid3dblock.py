#!/usr/bin/env python
# File: 2dgrid3dblock.py
# Name: D.Saravanan
# Date: 06/12/2024
# Script compute thread index with 2D grid of 3D blocks

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

kernel = """
#include <stdio.h>

/* 2D grid of 3D blocks */
__global__ void threadId_2D_3D() {
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) \
				   + (threadIdx.z * (blockDim.x * blockDim.y)) \
				   + (threadIdx.y * blockDim.x) + threadIdx.x;

	printf("blockId: %d = blockIdx.x: %d + blockIdx.y: %d * gridDim.x: %d\\n",\
			blockId, blockIdx.x, blockIdx.y, gridDim.x);
	printf("\\n");
	printf("threadId: %d = blockId: %d * (blockDim.x: %d * blockDim.y: %d "
		   "* blockDim.z: %d) + (threadIdx.z: %d * (blockDim.x: %d * blockDim.y: %d)) "
	       "+ (threadIdx.y: %d * blockDim.x: %d) + threadIdx.x: %d\\n", threadId,\
	       blockId, blockDim.x, blockDim.y, blockDim.z, threadIdx.z, blockDim.x,\
	       blockDim.y, threadIdx.y, blockDim.x, threadIdx.x);
}
"""

mod = SourceModule(kernel)
threadId_2D_3D = mod.get_function("threadId_2D_3D")
threadId_2D_3D(grid=(2,3,1), block=(4,2,2))

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
blockId: 2 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2
blockId: 2 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2
blockId: 2 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2
blockId: 2 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2
blockId: 2 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2
blockId: 2 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2
blockId: 2 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2
blockId: 2 = blockIdx.x: 0 + blockIdx.y: 1 * gridDim.x: 2
blockId: 0 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2
blockId: 0 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2
blockId: 0 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2
blockId: 0 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2
blockId: 0 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2
blockId: 0 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2
blockId: 0 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2
blockId: 0 = blockIdx.x: 0 + blockIdx.y: 0 * gridDim.x: 2
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
blockId: 5 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2
blockId: 5 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2
blockId: 5 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2
blockId: 5 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2
blockId: 5 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2
blockId: 5 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2
blockId: 5 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2
blockId: 5 = blockIdx.x: 1 + blockIdx.y: 2 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 1 = blockIdx.x: 1 + blockIdx.y: 0 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 3 = blockIdx.x: 1 + blockIdx.y: 1 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2
blockId: 4 = blockIdx.x: 0 + blockIdx.y: 2 * gridDim.x: 2


threadId: 80 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 0
threadId: 81 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 1
threadId: 82 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 2
threadId: 83 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 3
threadId: 84 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 0
threadId: 85 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 1
threadId: 86 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 2
threadId: 87 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 3
threadId: 88 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 0
threadId: 89 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 1
threadId: 90 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 2
threadId: 91 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 3
threadId: 92 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 0
threadId: 93 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 1
threadId: 94 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 2
threadId: 95 = blockId: 5 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 3
threadId: 32 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 0
threadId: 33 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 1
threadId: 34 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 2
threadId: 35 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 3
threadId: 36 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 0
threadId: 37 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 1
threadId: 38 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 2
threadId: 39 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 3
threadId: 40 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 0
threadId: 41 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 1
threadId: 42 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 2
threadId: 43 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 3
threadId: 44 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 0
threadId: 45 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 1
threadId: 46 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 2
threadId: 47 = blockId: 2 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 3
threadId: 16 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 0
threadId: 17 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 1
threadId: 18 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 2
threadId: 19 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 3
threadId: 20 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 0
threadId: 21 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 1
threadId: 22 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 2
threadId: 23 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 3
threadId: 24 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 0
threadId: 25 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 1
threadId: 26 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 2
threadId: 27 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 3
threadId: 28 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 0
threadId: 29 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 1
threadId: 30 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 2
threadId: 31 = blockId: 1 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 3
threadId: 48 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 0
threadId: 49 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 1
threadId: 50 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 2
threadId: 51 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 3
threadId: 52 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 0
threadId: 53 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 1
threadId: 54 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 2
threadId: 55 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 3
threadId: 56 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 0
threadId: 57 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 1
threadId: 58 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 2
threadId: 59 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 3
threadId: 60 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 0
threadId: 61 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 1
threadId: 62 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 2
threadId: 63 = blockId: 3 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 3
threadId: 64 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 0
threadId: 65 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 1
threadId: 66 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 2
threadId: 67 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 3
threadId: 68 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 0
threadId: 69 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 1
threadId: 70 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 2
threadId: 71 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 3
threadId: 72 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 0
threadId: 73 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 1
threadId: 74 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 2
threadId: 75 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 3
threadId: 76 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 0
threadId: 77 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 1
threadId: 78 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 2
threadId: 79 = blockId: 4 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 3
threadId: 0 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 0
threadId: 1 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 1
threadId: 2 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 2
threadId: 3 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 3
threadId: 4 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 0
threadId: 5 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 1
threadId: 6 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 2
threadId: 7 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 0 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 3
threadId: 8 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 0
threadId: 9 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 1
threadId: 10 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 2
threadId: 11 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 0 * blockDim.x: 4) + threadIdx.x: 3
threadId: 12 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 0
threadId: 13 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 1
threadId: 14 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 2
threadId: 15 = blockId: 0 * (blockDim.x: 4 * blockDim.y: 2 * blockDim.z: 2) + (threadIdx.z: 1 * (blockDim.x: 4 * blockDim.y: 2)) + (threadIdx.y: 1 * blockDim.x: 4) + threadIdx.x: 3

"""
