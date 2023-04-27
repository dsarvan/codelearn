#!/usr/bin/env python
# File: parallelarray.py
# Name: D.Saravanan
# Date: 28/10/2022

""" Script to pass explicit MPI datatypes
    and automatic MPI datatype discovery """

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# pass explicit MPI datatypes
if rank == 0:
	data = np.arange(1000, dtype='int')
	comm.Send([data, MPI.INT], dest=1, tag=11)

if rank == 1:
	data = np.empty(1000, dtype='int')
	comm.Recv([data, MPI.INT], source=0, tag=11)


# automatic MPI datatype discovery
if rank == 0:
	data = np.arange(1000, dtype=np.float64)
	comm.Send(data, dest=1, tag=21)

if rank == 1:
	data = np.empty(1000, dtype=np.float64)
	comm.Recv(data, source=0, tag=21)
