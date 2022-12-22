#!/usr/bin/env julia
# File: mpi4jl3.jl
# Name: D.Saravanan
# Date: 21/10/2022

""" Send multiple data items from one process to another with data tagging """

import MPI
MPI.Init()

function main()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    if rank == 0
        shared1 = Dict("d1" => 55, "d2" => 42)
        shared2 = Dict("d3" => 25, "d4" => 22)
        MPI.send(shared1, dest = 1, tag = 1, comm)
        MPI.send(shared2, dest = 1, tag = 2, comm)
        MPI.send(shared1, dest = 2, tag = 1, comm)
        MPI.send(shared2, dest = 2, tag = 2, comm)
        println("From rank $rank, we sent $shared1 and $shared2")

    elseif rank in (1, 2)
        receive1 = MPI.recv(source = 0, tag = 1, comm)
        receive2 = MPI.recv(source = 0, tag = 2, comm)
        println("On rank $rank, we received $receive1 and $receive2")

    else
        receive = 0.0
        println("On rank $rank, receive is $receive")
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
