#!/usr/bin/env python
# File: mpi4py6.py
# Name: D.Saravanan
# Date: 21/10/2022

""" Gathering - the main process gathers all the data processed by the other processes """

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data = [x for x in range(0, size)] if rank == 0 else None

data = comm.scatter(data, root=0)
print(f"Scattered data:")
print(f"rank: {rank}, data: {data}")

ndata = comm.gather(data, root=0)
if rank == 0:
    print(f"Gathered data:")
    print(f"rank: {rank}, data: {ndata}")
