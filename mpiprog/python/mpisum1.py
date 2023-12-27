#!/usr/bin/env python
# File: mpisum1.py
# Name: D.Saravanan
# Date: 03/12/2023

""" Script to initialize an array in rank 0 process, send data from
process 0 to all other processes in a communicator, assign values,
compute sum on each process and compute collective sum operation """

import numpy as np
from mpi4py import MPI


def main():
    """collective sum operation"""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    time = MPI.Wtime()

    N = 20000000  # array size

    # initialize array if rank = 0
    data = np.array([i * 1.0 for i in range(0, N)]) if rank == 0 else None

    workload = [N // size for _ in range(size)]
    for n in range(N % size):
        workload[n] += 1

    n = workload[rank]  # number of elements send to each process
    ndata = np.zeros(workload[rank])

    # send data from process 0 to all other processes in a communicator
    comm.Scatterv(data, ndata, root=0)

    del data

    # assign values and compute sum
    ndata = ndata + ndata
    nsum = np.sum(ndata)

    del ndata

    # collective computation sum operation
    tsum = comm.reduce(nsum, MPI.SUM, root=0)

    if rank == 0:  # print result
        print(f"Final sum = {tsum:e}")

    time = MPI.Wtime() - time
    print(f"Timing from rank {rank} is {time:0.6f} seconds")


if __name__ == "__main__":
    main()
