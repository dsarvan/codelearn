#!/usr/bin/env python
# File: trapezoidd.py
# Name: D.Saravanan
# Date: 09/03/2023

""" Script for trapezoid rule """

import numpy as np
from mpi4py import MPI


def f(x):
    return np.sin(x)


def trapezoidal(a, b, n, h):
    s = 0.0
    s += h * f(a)
    for i in range(1, n):
        s += 2.0 * h * f(a + i * h)
    s += h * f(b)
    return s / 2.0


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print(f"Number of ranks = {size}")

# print(f"rank = {rank}")

# Integral parameters
a = 0.0
b = np.pi
n = 100000000
h = (b - a) / n

dest = 0
total = -1.0
wload = n // size

la = a + rank * wload * h
lb = la + wload * h

integral = trapezoidal(la, lb, wload, h)

if rank == 0:
    total = integral
    for n in range(1, size):
        integral = comm.recv(source=n)
        print("PE ", rank, "<-", n, ",", integral, "\n")
        total = total + integral
else:
    print("PE ", rank, "->", dest, ",", integral, "\n")
    comm.send(integral, dest=0)


if rank == 0:
    print("**With n = ", n, ", trapezoids, ")
    print("**Final integral from", a, "to", b, "=", total, "\n")

MPI.Finalize()
