#!/usr/bin/env julia
# File: mpiprog.jl
# Name: D.Saravanan
# Date: 08/11/2022

""" MPI program working example """

import MPI

function main()
    
    # Initializes after program start
    MPI.Init()

    # MPI.COMM_WORLD is the communicator
    comm = MPI.COMM_WORLD

    # Gets the rank number of the process
    rank = MPI.Comm_rank(comm)

    # Gets the number of ranks in the program
    # determined by the mpiexec command
    size = MPI.Comm_size(comm)

    println("rank $rank of nproc $size\n")
    
    # Finalizes MPI to synchronize ranks and then exits
    MPI.Finalize()

end

main()
