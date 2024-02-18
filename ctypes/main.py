#!/usr/bin/env python
# File: main.py
# Name: D.Saravanan
# Date: 18/02/2024

""" Script to solve the laplace equation using Ctypes """

import ctypes

import numpy as np


def solve_equation(libc, u, n, dx):
    """solve the laplace equation"""

    nval, err = 0, 1

    while nval < 10000 and err > 1e-6:
        err = libc.time_step(u.ctypes.data_as(ctypes.c_void_p), n, n, dx, dx)
        nval += 1

    return err


def main():
    """main function"""

    # load the shared library into ctypes
    libc = ctypes.cdll.LoadLibrary("./laplace.so")

    libc.time_step.restype = ctypes.c_double
    libc.solve_equation.restype = ctypes.c_double

    n = ctypes.c_int(51)
    dx = ctypes.c_double(np.pi / 50)

    u = np.zeros((51, 51), dtype=np.float64)
    x = np.arange(0, np.pi + np.pi / 50, np.pi / 50, dtype=np.float64)

    u[0, :], u[50, :] = np.sin(x), np.sin(x)

    # computes the solution in Python
    solution1 = solve_equation(libc, u, n, dx)
    print(solution1)

    # computes the solution in C
    solution2 = libc.solve_equation(u.ctypes.data_as(ctypes.c_void_p), n, n, dx, dx)
    print(solution2)


if __name__ == "__main__":
    main()
