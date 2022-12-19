#!/usr/bin/env python
# File: parallelsum.py
# Name: D.Saravanan
# Date: 23/10/2022

""" To sum an array of numbers, we distribute the numbers among the processes
that compute the sum of a slice. The sums of the slices are sent to process 0
that computes the total sum. """

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 10000  # number of data to be send to processes

if rank == 0:
    data = np.arange(N * size, dtype="int")

    """ distributing slices """
    for n in range(1, size):
        slice = data[n * N : (n + 1) * N]
        comm.send(slice, dest=n)

    ndata = data[0:N]

else:
    ndata = np.empty(N, dtype="int")
    ndata = comm.recv(source=0)

sval = sum(ndata)
print(f"{rank} has data, {ndata}, sum = {sval}")


""" collecting the sums of the slices """
tsum = np.zeros(size, dtype="int")

if rank > 0:
    comm.send(sval, dest=0)
else:
    tsum[0] = sval

    for n in range(1, size):
        tsum[n] = comm.recv(source=n)

    tsval = sum(tsum)
    print(f"Total sum value: {tsval}")
