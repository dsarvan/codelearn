#!/usr/bin/env python
# File: vectorsum.py
# Name: D.Saravanan
# Date: 28/11/2024
# Script computes the sum of two arrays on the GPU using PyCUDA

import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void vectorSum(float *a, float *b, float *c) {
	const int idx = threadIdx.x;
	c[idx] = a[idx] + b[idx];
}
""")

vectorSum = mod.get_function("vectorSum")

a = np.random.randn(400).astype(np.float32)
b = np.random.randn(400).astype(np.float32)
c = np.zeros_like(a)

vectorSum(drv.In(a), drv.In(b), drv.Out(c), grid=(1,1), block=(400,1,1))
np.testing.assert_array_equal(c, a+b, strict=True)
