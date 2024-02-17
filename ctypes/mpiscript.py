#!/usr/bin/env python
# File: mpiscript.py
# Name: D.Saravanan
# Date: 16/02/2024

""" Script to wrapping function from matmult.c using ctypes """

import ctypes
import sys


def main():
    """wrapping using shared mpi libraries with ctypes"""

    # load the shared mpi library into ctypes
    mpic = ctypes.cdll.LoadLibrary("./matmult.so")

    argc = len(sys.argv)
    argv = sys.argv
    argv = (ctypes.c_wchar_p * argc)(*argv)

    nrow, nval, ncol = 10062, 10015, 1007

    mpic.matmult.restype = ctypes.POINTER(ctypes.c_double)
    mpic.matmult.argtypes = [
        ctypes.c_int,
        ctypes.c_wchar_p * argc,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]

    matrix = mpic.matmult(argc, argv, nrow, nval, ncol)

    print("Here is the result matrix:")
    for i in range(nrow):
        print()
        for j in range(ncol):
            print(f"{matrix[i*ncol + j]:6.2e}", end=" ")
    print()


if __name__ == "__main__":
    main()
