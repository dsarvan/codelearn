#!/usr/bin/env python
# File: vectorinc.py
# Name: D.Saravanan
# Date: 01/04/2023

""" Script to parallelize for loops """

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void vecinc(int *a, int N) {
	int i = threadIdx.x;

	if (i < N)
		a[i] = a[i] + 1;
}
""")

if __name__ == "__main__":

	N = 1000000000

	# allocate input vector h_a in host memory
	h_a = np.zeros(N).astype(np.float32)

	# allocate vector in device memory
	d_a = cuda.mem_alloc(h_a.nbytes)

	# copy vector from host memory to device memory
	cuda.memcpy_htod(d_a, h_a)

	func = mod.get_function("vecinc")
	func(np.array(N), d_a, block=(1,1,1), grid=(N,1,1))
