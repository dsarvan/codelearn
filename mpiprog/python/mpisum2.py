#!/usr/bin/env python
# File: mpisum2.py
# Name: D.Saravanan
# Date: 03/12/2023

""" Script to initialize an array on each processes in a communicator,
assign values, compute sum on each process and compute collective sum operation """

import numpy as np
from mpi4py import MPI


def main():
    """collective sum operation"""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    time = MPI.Wtime()

    N = 20000000  # array size

    workload = [N // size for _ in range(size)]
    for n in range(N % size):
        workload[n] += 1

    sload = 0
    for n in range(rank):
        sload += workload[n]

    n = workload[rank]  # number of elements send to each process

    # initialize array on each process
    ndata = np.array([(i + sload) * 1.0 for i in range(n)])

    # assign values and compute sum
    ndata = ndata + ndata
    nsum = np.sum(ndata)

    del ndata

    # collective computation sum operation
    tsum = comm.reduce(nsum, MPI.SUM, root=0)

    if rank == 0:
        print(f"Final sum = {tsum:e}")

    time = MPI.Wtime() - time
    print(f"Timing from rank {rank} is {time:0.6f} seconds")


if __name__ == "__main__":
    main()
