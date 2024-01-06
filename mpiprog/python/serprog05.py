#!/usr/bin/env python
# File: serprog05.py
# Name: D.Saravanan
# Date: 04/12/2023

""" Script to initialize matrices, assign values and compute matrix multiplication """

import numpy as np


def main():
    """matrix multiplication"""

    nrow, nval, ncol = 10062, 10015, 1007

    print("Starting serial matrix multiplication ...")
    print(f"Using matrix sizes A[{nrow}][{nval}], B[{nval}][{ncol}], C[{nrow}][{ncol}]")

    print("Initializing matrices ...")
    A = np.array([[i + j for j in range(nval)] for i in range(nrow)], dtype=float)
    B = np.array([[i * j for j in range(ncol)] for i in range(nval)], dtype=float)

    # matrix multiplication
    print("Performing matrix multiplication ...")
    C = A @ B

    # print matrix result
    print("Here is the result matrix:")
    print(C)


if __name__ == "__main__":
    main()
