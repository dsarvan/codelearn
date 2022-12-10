#!/usr/bin/env python
# File: mpi4py2.py
# Name: D.Saravanan
# Date: 21/10/2022

""" Using send() and receive() for data transfer between processes is the
simplest form of communication between processes. We can achieve one-to-one
communication with this. """

from mpi4py import MPI

def main():

	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()

	value = 3.1415

	if rank == 0:
		data = value
		comm.send(data, dest=1)
		comm.send(data, dest=2)
		print(f"From rank {rank}, we sent {data}")

	elif rank in (1, 2):
		data = comm.recv(source=0)
		print(f"On rank {rank}, we received {data}")

	else:
		data = 0.0
		print(f"On rank {rank}, data is {data}")

if __name__ == "__main__":
	main()
