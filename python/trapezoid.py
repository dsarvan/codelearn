#!/usr/bin/env python
# File: trapezoid.py
# Name: D.Saravanan
# Date: 08/03/2023

""" Script for trapezoid rule """

from typing import Optional

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def trapezoid(a: float, b: float, N: int) -> Optional[float]:
    """numerical integration"""

    wload = [N // size for _ in range(size)]
    for n in range(N % size):
        wload[n] += 1

    # trapezoid base length (the same for all processes)
    h = (b - a) / N

    la = a + rank * wload[rank] * h
    lb = la + wload[rank] * h

    x = np.linspace(la, lb, wload[rank] + 1)
    f = np.sin(x)  # integration function

    trap = (h / 2) * (f[0] + 2 * np.sum(f[1 : wload[rank]]) + f[wload[rank]])
    trap = comm.reduce(trap, op=MPI.SUM, root=0)

    if rank == 0:
        return trap

    return None


if __name__ == "__main__":
    # integral parameters
    a: float = 0.0  # left endpoint
    b: float = np.pi  # right endpoint
    n: int = 100000000  # number of trapezoids

    integral: Optional[float] = trapezoid(a, b, n)
    if rank == 0:
        print(integral)
