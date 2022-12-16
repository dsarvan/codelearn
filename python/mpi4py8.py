#!/usr/bin/env python
# File: mpi4py8.py
# Name: D.Saravanan
# Date: 21/10/2022

""" Broadcasting a NumPy array """

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
	data = np.arange(100, dtype=np.int32)
else:
	data = np.empty(100, dtype=np.int32)

comm.Bcast(data, root=0)

for n in range(100):
	assert data[n] == n
