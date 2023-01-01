#!/usr/bin/env python
# File: parallelrun3.py
# Name: D.Saravanan
# Date: 23/10/2022

""" Script to add two arrays and average the result """

from mpi4py import MPI
import numpy as np


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = 10000000

    # determine the workload of each rank
    workloads = [N // size for _ in range(size)]

    for n in range(N % size):
        workloads[n] += 1

    """
	sload and eload represent the range
	over which each rank will perform mathematical
	operations on the arrays.

	"""
    sload = 0
    for n in range(rank):
        sload += workloads[n]
    eload = sload + workloads[rank]

    # initialize a
    stime = MPI.Wtime()
    a = np.ones(workloads[rank], dtype=np.int64)
    etime = MPI.Wtime()
    if rank == 0:
        print(f"Initialize a time: {etime - stime}")

    # initialize b
    stime = MPI.Wtime()
    b = np.arange(sload, eload, dtype=np.int64)
    b = b + 1
    etime = MPI.Wtime()
    if rank == 0:
        print(f"Initialize b time: {etime - stime}")

    # add the two arrays
    stime = MPI.Wtime()
    a = a + b
    etime = MPI.Wtime()
    if rank == 0:
        print(f"Add arrays time: {etime - stime}")

    sval = np.sum(a)

    # average the result
    stime = MPI.Wtime()
    tsum = np.zeros(size, dtype=np.int64)
    if rank > 0:
        comm.send(sval, dest=0)
    else:
        tsum[0] = sval
        for n in range(1, size):
            tsum[n] = comm.recv(source=n)

        average = np.sum(tsum) / N

    etime = MPI.Wtime()

    if rank == 0:
        print(f"Average result time: {etime - stime}")
        print(f"Average: {average}")


if __name__ == "__main__":
    main()
