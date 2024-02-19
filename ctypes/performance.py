#!/usr/bin/env python
# File: performance.py
# Name: D.Saravanan
# Date: 19/02/2024

""" Script to call compute and computepy function """

import ctypes
import time

import numpy as np

import computepy


def main():
    """ main function """

    nval = 400000000

    np.random.seed(42)
    arr = np.random.random(nval)

    stime = time.time()
    sum_t = computepy.sumval(arr)
    etime = time.time() - stime

    print(f"Sum: {sum_t:.1f} in {etime:.3f} seconds")

    libc = ctypes.cdll.LoadLibrary("./compute.so")
    libc.sumval.restype = ctypes.c_double

    stime = time.time()
    sum_t = libc.sumval(nval, ctypes.c_void_p(arr.ctypes.data))
    etime = time.time() - stime

    print(f"Sum: {sum_t:.1f} in {etime:.3f} seconds")


if __name__ == "__main__":
    main()
