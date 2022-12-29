#!/usr/bin/env python
# File: parallelrun1.py
# Name: D.Saravanan
# Date: 23/10/2022

""" Script to add two arrays and average the result """

from mpi4py import MPI
import numpy as np

def main():

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()

	N = 10000

	# initialize a
	stime = MPI.Wtime()
	a = np.ones(N)
	etime = MPI.Wtime()
	if rank == 0:
		print(f"Initialize a time: {etime - stime}")

	# initialize b
	stime = MPI.Wtime()
	b = np.arange(N)
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

	# average the result
	stime = MPI.Wtime()
	average = sum(a)/N
	etime = MPI.Wtime()
	if rank == 0:
		print(f"Average result time: {etime - stime}")
		print(f"Average: {average}")


if __name__ == "__main__":
	main()
