#!/usr/bin/env python
# File: problem.py
# Name: D.Saravanan
# Date: 20/02/2024

""" Script to allocate dynamic memory with calloc from Python using Ctypes """

import ctypes


def main():
    """Note that using calloc and free directly in Python
    is not recommended as it can lead to memory leaks and
    segmentation faults if used improperly. It is better
    to use Python's built-in data types and objects."""

    nval = 10

    # load libc.so.6 on Linux
    libc = ctypes.cdll.LoadLibrary("libc.so.6")

    # calloc return type
    libc.calloc.restype = ctypes.POINTER(ctypes.c_double)

    # libc.calloc() function to allocate memory,
    # specifying the size of the memory to be allocated
    # libc.calloc() function returns a pointer to the allocated memory
    x = libc.calloc(nval, ctypes.sizeof(ctypes.c_double))

    # initialize
    for i in range(nval):
        x[i] = 2 * i

    # print elements
    for i in range(nval):
        print(x[i])

    # libc.free() function to release the allocated memory
    libc.free(x)


if __name__ == "__main__":
    main()
