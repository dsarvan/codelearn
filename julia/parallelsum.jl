#!/usr/bin/env julia
# File: parallelsum.jl
# Name: D.Saravanan
# Date: 23/10/2022

""" To sum an array of numbers, we distribute the numbers among the processes
that compute the sum of a slice. The sums of the slices are sent to process 0
that computes the total sum. """

import MPI
MPI.Init()

function main()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    N = 10000   # number of data to be send to processes

    if rank == 0
        data = 1:1:(N*size)

        """ distributing slices """
        for n in range(1, size - 1)
            slice = data[(n*N)+1:(n+1)*N]
            MPI.send(slice, dest = n, comm)
        end

        ndata = data[1:N]

    else
        ndata = zeros(Int, N)
        ndata = MPI.recv(source = 0, comm)
    end

    sval = sum(ndata)
    println("$rank has data, $ndata, sum = $sval")


    """ collecting the sums of the slices """
    tsum = zeros(Int, size)

    if rank > 0
        MPI.send(sval, dest = 0, comm)
    else
        tsum[1] = sval

        for n in range(2, size)
            tsum[n] = MPI.recv(source = (n - 1), comm)
        end

        tsval = sum(tsum)
        println("Total sum values: $tsval")
    end

    MPI.Barrier(comm)
    MPI.Finalize()

end

main()
