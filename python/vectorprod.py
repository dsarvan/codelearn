#!/usr/bin/env python
# File: vectorprod.py
# Name: D.Saravanan
# Date: 01/04/2023

""" Script compute the product of two arrays """

import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void vecprod(float *a, float *b, float *c) {
	int i = threadIdx.x;
	c[i] = a[i] * b[i];
}
""")

if __name__ == "__main__":
    N = 1024  # Maximum threads per block (Tesla T4)

    a = np.random.rand(N).astype(np.float32)
    b = np.random.rand(N).astype(np.float32)
    c = np.zeros_like(a)

    func = mod.get_function("vecprod")
    func(drv.In(a), drv.In(b), drv.Out(c), block=(N, 1, 1), grid=(1, 1, 1))

    print(c - a * b)
