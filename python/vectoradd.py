#!/usr/bin/env python
# File: vectoradd.py
# Name: D.Saravanan
# Date: 01/04/2023

""" Script compute the sum of two arrays """

import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void vecadd(float *a, float *b, float *c) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}
""")

if __name__ == "__main__":
    N = 1024  # Maxmimum threads per block (Tesla T4)

    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    c = np.zeros_like(a)

    func = mod.get_function("vecadd")
    func(drv.In(a), drv.In(b), drv.Out(c), block=(N, 1, 1), grid=(1, 1))

    print(c - (a + b))
