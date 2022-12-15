#!/usr/bin/env python
# File: mpi4py7.py
# Name: D.Saravanan
# Date: 21/10/2022

""" Passing MPI datatype explicitly and Automatic MPI datatype discovery """

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# passing MPI datatypes explicitly
if rank == 0:
	data = np.arange(1000, dtype="int")
	comm.Send([data, MPI.INT], dest=1, tag=11)
elif rank == 1:
	data = np.empty(1000, dtype="int")
	comm.Recv([data, MPI.INT], source=0, tag=11)


# automatic MPI datatype discovery
if rank == 0:
	data = np.arange(1000, dtype=np.int32)
	comm.Send(data, dest=1, tag=21)
elif rank == 1:
	data = np.empty(1000, dtype=np.int32)
	comm.Recv(data, source=0, tag=21)
