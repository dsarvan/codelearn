#!/usr/bin/env python
# File: mpi4py10.py
# Name: D.Saravanan
# Date: 21/10/2022

""" Gathering NumPy arrays """

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sendbuf = np.zeros(100, dtype=np.int32) + rank
recvbuf = None

if rank == 0:
	recvbuf = np.empty([size, 100], dtype=np.int32)

comm.Gather(sendbuf, recvbuf, root=0)

if rank == 0:
	for n in range(size):
		assert np.allclose(recvbuf[n,:] , n)
