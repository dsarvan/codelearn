#!/usr/bin/env python
# File: simpson.py
# Name: D.Saravanan
# Date: 13/03/2023

""" Script for Simpson's rule """


import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def simpson(a, b, N):
    """Note that to use Simpson's rule, you
    must have an even number of intervals and
    therefore, an odd number of grid points."""

    wload = [N // size for _ in range(size)]
    for n in range(N % size):
        wload[n] += 1

    h = (b - a) / N

    la = a + rank * wload[rank] * h
    lb = la + wload[rank] * h

    # odd number of grid points
    x = np.linspace(la, lb, wload[rank] + 1)
    f = np.sin(x)  # integration function

    simp = (h / 3) * (
        f[0]
        + 2 * np.sum(f[0 : wload[rank] - 1 : 2])
        + 4 * np.sum(f[1 : wload[rank] : 2])
        + f[wload[rank]]
    )
    simp = comm.reduce(simp, op=MPI.SUM, root=0)

    if rank == 0:
        return simp


if __name__ == "__main__":
    # integral parameters
    a = 0.0  # left endpoint
    b = np.pi  # right endpoint
    n = 100000000  # number of intervals

    integral = simpson(a, b, n)
    if rank == 0:
        print(integral)
