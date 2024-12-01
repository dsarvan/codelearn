#!/usr/bin/env python
# File: 1dgrid1dblock.py
# Name: D.Saravanan
# Date: 01/12/2024
# Script compute thread index with 1D grid of 1D blocks

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

kernel = """
#include <stdio.h>

/* 1D grid of 1D blocks */
__global__ void threadId_1D_1D() {
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	printf("threadId: %d = blockIdx.x: %d * blockDim.x: %d + threadIdx.x: %d\\n",\
	        threadId, blockIdx.x, blockDim.x, threadIdx.x);
}
"""

mod = SourceModule(kernel)
threadId_1D_1D = mod.get_function("threadId_1D_1D")
threadId_1D_1D(grid=(2,1,1), block=(4,1,1)) # kernel launch

"""
Output:

threadId: 0 = blockIdx.x: 0 * blockDim.x: 4 + threadIdx.x: 0
threadId: 1 = blockIdx.x: 0 * blockDim.x: 4 + threadIdx.x: 1
threadId: 2 = blockIdx.x: 0 * blockDim.x: 4 + threadIdx.x: 2
threadId: 3 = blockIdx.x: 0 * blockDim.x: 4 + threadIdx.x: 3
threadId: 4 = blockIdx.x: 1 * blockDim.x: 4 + threadIdx.x: 0
threadId: 5 = blockIdx.x: 1 * blockDim.x: 4 + threadIdx.x: 1
threadId: 6 = blockIdx.x: 1 * blockDim.x: 4 + threadIdx.x: 2
threadId: 7 = blockIdx.x: 1 * blockDim.x: 4 + threadIdx.x: 3

"""
