#!/usr/bin/env python
# File: paralleltrap.py
# Name: D.Saravanan
# Date: 01/11/2022

""" Script for parallel computation of integral by trapezoidal rule """

from mpi4py import MPI
import numpy as np

@profile
def main():

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()

	# end points of interval
	a, b = 0, 1

	# function to be integrated
	f = lambda x: np.exp(-x**2)

	# exact value of the integral
	exact = 0.746824132812427025399

	# number of different set of grids
	Ngrids = 10

	h = np.zeros(Ngrids, dtype=np.float64)

	trap = np.zeros(Ngrids, dtype=np.float64)

	workloads = [Ngrids//size for _ in range(size)]
	for n in range(Ngrids%size):
		workloads[n] += 1

	sload = 0
	for n in range(rank):
		sload += workloads[n]
	eload = sload + workloads[rank]

	for k in range(0, Ngrids):

		N = 10*2**(k+2)
		h[k] = (b - a)/(N - 1)
		x = np.linspace(a, b, N)
		
		workloads = [len(x)//size for _ in range(size)]
		for n in range(len(x) % size):
			workloads[n] += 1
		
		sload = 0
		for n in range(rank):
			sload += workloads[n]
		eload = sload + workloads[rank]

		sval = np.sum(f(x[sload:eload]))
		tsum = np.zeros(size, dtype=np.float64)
		if rank > 0:
			comm.send(sval, dest=0)
		else:
			tsum[0] = sval
			for n in range(1, size):
				tsum[n] = comm.recv(source=n)

		trap[k] = h[k]*(np.sum(tsum) - (f(a) + f(b))/2)

	if rank == 0:
		print(trap)

if __name__ == "__main__":
	main()
