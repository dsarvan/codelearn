#!/usr/bin/env python
# File: mpiprog01.py
# Name: D.Saravanan
# Date: 21/10/2022

""" Script to print the rank of the current process, the number of MPI
processes and the hostname on which the current process is running """

from mpi4py import MPI

# MPI.COMM_WORLD is the communicator.
# It is used for all MPI communication
# between the processes running on the
# processes of the cluster.
comm = MPI.COMM_WORLD

# the rank of the current process
rank = comm.Get_rank()

# the number of MPI processes
size = comm.Get_size()

# the hostname on which the current process is running
name = MPI.Get_processor_name()

print(f"rank {rank} of nproc {size} on {name}")
