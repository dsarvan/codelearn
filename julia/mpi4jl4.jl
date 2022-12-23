#!/usr/bin/env julia
# File: mpi4jl4.jl
# Name: D.Saravanan
# Date: 21/10/2022

""" Broadcasting - data is sent from a single process to all the other processes """

import MPI
MPI.Init()

function main()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    if rank == 0
        data = Dict("d1" => 55, "d2" => 42)
    else
        data = 0
    end

    data = MPI.bcast(data, root = 0, comm)
    println("rank: $rank, data: $data")

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
