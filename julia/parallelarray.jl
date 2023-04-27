#!/usr/bin/env julia
# File: parallelarray.jl
# Name: D.Saravanan
# Date: 28/10/2022

""" Script to pass explicit MPI datatypes
    and automatic MPI datatype discovery """

import MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

# pass explicit MPI datatypes
if rank == 0
    data = 1:10
    MPI.Send(data, dest = 1, tag = 11, comm)
end

if rank == 1
    data = MPI.Recv(Int, source = 0, tag = 11, comm)
end

# automatic MPI datatype discovery
if rank == 0
    data = 1:1:10
    MPI.Send(data, dest=1, tag=21, comm)
end

if rank == 1
    data = MPI.Recv(source=0, tag=21, comm)
end

MPI.Barrier(comm)
MPI.Finalize()
