#!/usr/bin/env julia
# File: broadcast.jl
# Name: D.Saravanan
# Date: 07/03/2023

""" Script to broadcast data to all processes """

import MPI
MPI.Init()

function main()
    """broadcast data to all processes"""
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    if rank == 0
        data = Dict(
            "key1" => [10, 10.1, 10 + 11im],
            "key2" => ["mpi-julia", "julia"],
            "key3" => Array([1, 2, 3]),
        )
    else
        data = 0
    end

    # broadcast data to all processes
    data = MPI.bcast(data, root = 0, comm)

    if rank == 0
        println("bcast finished")
    end

    println("data on rank $rank of size $size is $data")

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
