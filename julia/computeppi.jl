#!/usr/bin/env julia
# File: computeppi.jl
# Name: D.Saravanan
# Date: 06/03/2023

""" Script to compute pi with MPI """

import MPI
MPI.Init()

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const size = MPI.Comm_size(comm)

function computepi(N)
    """ compute pi """
    dx = 1 / N # step size

    workloads = [trunc(Int, N / size) for _ in range(1, size)]
    for n in range(1, N % size)
        workloads[n] += 1
    end

    sload = 0
    for n in range(1, rank)
        sload += workloads[n+1]
    end
    eload = sload + workloads[rank+1]

    x = i -> (i + 0.5) * dx
    pi = sum(map(i -> 4.0 / (1 + x(i)^2) * dx, sload+1:eload))
    pi = MPI.Reduce(pi, +, root = 0, comm)

    if rank == 0
        return pi
    end

end


function main()
    ns = 100000000
    pi = computepi(ns)

    if rank == 0
        println(pi)
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
