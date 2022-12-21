#!/usr/bin/env julia
# File: mpi4jl2.jl
# Name: D.Saravanan
# Date: 21/10/2022

""" Using Send() and Recv!() for data transfer between processes is the 
simplest form of communication between processes. We can achieve one-to-one 
communication with this. """

import MPI
MPI.Init()

function main()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    value = 3.1415

    if rank == 0
        data = value
        MPI.send(data, dest = 1, comm)
        MPI.send(data, dest = 2, comm)
        println("From rank $rank, we send $data")

    elseif rank in (1, 2)
        data = MPI.recv(source = 0, comm)
        println("On rank $rank, we received $data")

    else
        data = 0.0
        println("On rank $rank, data is $data")
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
