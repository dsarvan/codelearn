#!/usr/bin/env python
# File: mpi4py9.py
# Name: D.Saravanan
# Date: 21/10/2022

""" Scattering NumPy arrays """

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sendbuf = None
if rank == 0:
	sendbuf = np.empty([size, 100], dtype=np.int32)
	sendbuf.T[:,:] = range(size)

recvbuf = np.empty(100, dtype=np.int32)

comm.Scatter(sendbuf, recvbuf, root=0)
assert np.allclose(recvbuf, rank)
