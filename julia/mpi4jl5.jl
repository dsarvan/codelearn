#!/usr/bin/env julia
# File: mpi4jl5.jl
# Name: D.Saravanan
# Date: 21/10/2022

""" Scattering - send the different chunks of the data to all the processes """

import MPI
MPI.Init()

function main()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    if rank == 0
        data = [x for x = 0:(size-1)]
    else
        data = 0
    end

    data = MPI.Scatter(data, Int32, root = 0, comm)
    println("rank: $rank, data: $data")

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
