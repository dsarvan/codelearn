#!/usr/bin/env python
# File: gather.py
# Name: D.Saravanan
# Date: 26/11/2023

""" Script that sends data from one process to all other processes in a communicator """

import numpy as np
from mpi4py import MPI


def main():
    """scattering numpy arrays"""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    time = MPI.Wtime()

    N = 200  # array size

    # initialize array if rank = 0
    data = np.array([i * 1.0 for i in range(0, N)]) if rank == 0 else None

    workload = [N // size for _ in range(size)]
    for n in range(N % size):
        workload[n] += 1

    n = workload[rank]  # number of elements send to each process
    ndata = np.zeros(workload[rank])

    # send data from process 0 to all other processes in a communicator
    comm.Scatter(data, ndata, root=0)

    del data

    # print data from all process
    for i in range(n - 5, n):
        print(f"rank = {rank}   data[{i}] = {ndata[i]}")

    del ndata

    time = MPI.Wtime() - time

    print(f"Timing from rank {rank} is {time:0.6f} seconds")


if __name__ == "__main__":
    main()
