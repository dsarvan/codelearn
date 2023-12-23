#!/usr/bin/env python
# File: gather.py
# Name: D.Saravanan
# Date: 26/11/2023

""" Script that receives data to process 0 from all other processes in a communicator """

import numpy as np
from mpi4py import MPI


def main():
    """gathering numpy arrays"""

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

    # assign values
    ndata = 2 * ndata

    print(f"Scatter data (process = {rank}):")
    for i in range(n - 5, n):
        print(f"rank = {rank}   data[{i}] = {ndata[i]}")

    # receive data to process 0 from all other processes in a communicator
    comm.Gather(ndata, data, root=0)

    del ndata

    if rank == 0:
        print("Gather data (process = 0)")
        for i in range(N - 5, N):
            print(f"rank = {rank}   data[{i}] = {data[i]}")

    del data

    time = MPI.Wtime() - time

    print(f"Timing from rank {rank} is {time:0.6f} seconds")


if __name__ == "__main__":
    main()
