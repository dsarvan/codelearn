#!/usr/bin/env julia
# File: sendrecv.jl
# Name: D.Saravanan
# Date: 08/03/2023

""" Script to create an array in rank 0 process and
send the first part of the array to the rank 1 process
and the second part of the array to the rank 2 process """

using MPI
MPI.Init()

function main()
    """ point-to-point communication """

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    if rank == 0
        data = range(0, 9)
        MPI.send(data[1:4], dest = 1, tag = 11, comm)
        MPI.send(data[6:9], dest = 2, tag = 12, comm)
        println("Rank $rank data is $data")
    elseif rank == 1
        data = MPI.recv(source = 0, tag = 11, comm)
        println("Rank $rank received data $data")
    elseif rank == 2
        data = MPI.recv(source = 0, tag = 12, comm)
        println("Rank $rank received data $data")
    else
        data = 0
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
