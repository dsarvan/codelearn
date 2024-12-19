#!/usr/bin/env python
# File: matrixmul.py
# Name: D.Saravanan
# Date: 20/12/2024
# Script to initialize matrices, assign values and compute matrix multiplication

import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule


mod = SourceModule("""
__global__ void matrixMultiply(int m, int v, int n, float *A, float *B, float *C) {
	const int idx = n * threadIdx.x + threadIdx.y;

	for(int k = 0; k < v; k++)
		C[idx] += A[v*threadIdx.x+k] * B[threadIdx.y+k*n];
}
""")


def main():
	"""Multiply an nrow-by-nval matrix with an nval-by-ncol matrix
	with a two dimensional grid of nrow x ncol threads."""

	nrow, nval, ncol = 30, 40, 10

	nr = np.int32(nrow)
	nv = np.int32(nval)
	nc = np.int32(ncol)

	A = np.random.randint(2, size=(nrow, nval))
	B = np.random.randint(2, size=(nval, ncol))
	C = np.zeros((nrow, ncol), dtype=np.float32)

	A = A.astype(np.float32)
	B = B.astype(np.float32)

	matrixMultiply = mod.get_function("matrixMultiply")
	matrixMultiply(nr, nv, nc, drv.In(A), drv.In(B), drv.Out(C), grid=(1, 1), block=(nrow, ncol, 1))

	np.testing.assert_array_equal(C, np.matmul(A, B), strict=True)


if __name__ == "__main__":
	main()
