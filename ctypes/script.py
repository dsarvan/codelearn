#!/usr/bin/env python
# File: script.py
# Name: D.Saravanan
# Date: 11/02/2024

""" Script to wrapping function from program.c using ctypes """

import ctypes

import numpy as np


def main():
    """wrapping using shared libraries with ctypes"""

    # load the shared library into ctypes
    libc = ctypes.cdll.LoadLibrary("./program.so")

    # call the c function compute from the library
    counter = libc.compute(10)
    print(counter)

    # call the c function multiply from the library
    libc.multiply.restype = ctypes.c_double
    libc.multiply.argtypes = [ctypes.c_int, ctypes.c_double]
    rval = libc.multiply(10, 9.8)
    print(rval)

    # call the c function mean from the library
    arr = [10, 20, 30, 40, 50]  # define the array in Python
    nval = len(arr)  # length of the array
    arr = (ctypes.c_long * nval)(*arr)

    libc.mean.restype = ctypes.c_double
    libc.mean.argtypes = [ctypes.c_long * nval, ctypes.c_long]
    mean = libc.mean(arr, nval)
    print(mean)

    # call the c function matrixC from the libaray
    libc.matrixC.restype = ctypes.POINTER(ctypes.c_double)
    libc.matrixC.argtypes = [ctypes.c_int, ctypes.c_int]

    mrow, mcol = 4, 4
    result = libc.matrixC(mrow, mcol)

    for n in range(mrow * mcol):
        print(result[n])

    # call the c function simulation from the library
    ex = np.zeros(200, dtype=np.float64)
    hy = np.zeros(200, dtype=np.float64)

    arr_1d_double = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags="CONTIGUOUS")

    libc.simulation.argtypes = [arr_1d_double, arr_1d_double]
    libc.simulation(ex, hy)
    print(hy)


if __name__ == "__main__":
    main()
