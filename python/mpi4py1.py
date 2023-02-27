#!/usr/bin/env python
# File: mpi4py1.py
# Name: D.Saravanan
# Date: 21/10/2022

""" Script to print the rank of the current process, the number of MPI 
processes and the hostname on which the current process is running """

import sys
from mpi4py import MPI

# MPI.COMM_WORLD is the communicator.
# It is used for all MPI communication
# between the processes running on the
# processes of the cluster.
comm = MPI.COMM_WORLD

# the rank of the current process
rank = comm.Get_rank()
sys.stdout.write("rank: %d\n" % (rank))

# the number of MPI processes
size = comm.Get_size()
sys.stdout.write("size: %d\n" % (size))

# the hostname on which the current process is running
name = MPI.Get_processor_name()

if rank == 0:
    sys.stdout.write("Doing the task of rank 0\n")

if rank == 1:
    sys.stdout.write("Doing the task of rank 1\n")
