#!/usr/bin/env julia
# File: mpi4jl1.jl
# Name: D.Saravanan
# Date: 21/10/2022

""" Script to print the rank of the current process, the number of MPI 
processes and the hostname on which the current process is running """

import MPI
MPI.Init()

using Printf

# MPI.COMM_WORLD is the communicator.
# It is used for all MPI communication
# between the processes running on the
# processes of the cluster.
comm = MPI.COMM_WORLD

# the rank of the process in the particular communicator's group
rank = MPI.Comm_rank(comm)
@printf("rank: %d\n", rank)

# the number of processes involved in communicator
size = MPI.Comm_size(comm)
@printf("size: %d\n", size)

#name = MPI.processor_name(comm)

if rank == 0
    @printf("Doing the task of rank 0\n")
end

if rank == 1
    @printf("Doing the task of rank 1\n")
end
