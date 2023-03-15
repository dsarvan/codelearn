#!/usr/bin/env julia
# File: simpson.jl
# Name: D.Saravanan
# Date: 14/03/2023

""" Script for Simpson's rule """

using MPI
MPI.Init()

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const size = MPI.Comm_size(comm)


function simpson(a, b, N)
    """Note that to use Simpson's rule, you
    must have an even number of intervals and
    therefore, an odd number of grid points."""

    wload = [trunc(Int, N / size) for _ in range(1, size)]
    for n in range(1, N % size)
        wload[n] += 1
    end

    h = (b - a) / N

    la = a + rank * wload[rank+1] * h
    lb = la + wload[rank+1] * h

    # odd number of grid points
    x = LinRange(la, lb, wload[rank+1] + 1)
    f = sin.(x)  # integration function

    simp = (h / 3) * (
            f[1] +
            2 * sum(f[1:2:wload[rank+1]-1]) +
            4 * sum(f[2:2:wload[rank+1]]) +
            f[wload[rank+1]+1]
        )
    simp = MPI.Reduce(simp, +, root = 0, comm)

    if rank == 0
        return simp
    end
end


function main()
    # integral parameters
    a = 0.0  # left endpoint
    b = pi  # right endpoint
    n = 100000000  # number of intervals

    integral = simpson(a, b, n)
    if rank == 0
        println(integral)
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
