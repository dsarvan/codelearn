#!/usr/bin/env python
# File: mpi4py4.py
# Name: D.Saravanan
# Date: 21/10/2022

""" Broadcasting - data is sent from a single process to all the other processes """

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    data = {"d1": 55, "d2": 42}
else:
    data = None

data = comm.bcast(data, root=0)
print(f"rank: {rank}, data: {data}")
