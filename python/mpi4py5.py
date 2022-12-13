#!/usr/bin/env python
# File: mpi4py5.py
# Name: D.Saravanan
# Date: 21/10/2022

""" Scattering - send the different chunks of the data to all the processes """

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data = [x for x in range(0, size)] if rank == 0 else None

data = comm.scatter(data, root=0)
print(f"rank: {rank}, data: {data}")
