#!/usr/bin/env julia
# File: trapezoid.jl
# Name: D.Saravanan
# Date: 08/03/2023

""" Script for trapezoid rule """

using MPI
MPI.Init()

const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const size = MPI.Comm_size(comm)


function trapezoid(a, b, N)
    """numerical integration"""

    wload = [trunc(Int, N / size) for _ in range(1, size)]
    for n in range(1, N % size)
        wload[n] += 1
    end

    # trapezoid base length (the same for all processes)
    h = (b - a) / N

    la = a + rank * wload[rank+1] * h
    lb = la + wload[rank+1] * h

    x = LinRange(la, lb, wload[rank+1] + 1)
    f = sin.(x)  # integration function

    trap = (h/2) * (f[1] + 2 * sum(f[2:wload[rank+1]]) + f[wload[rank+1]+1])
    trap = MPI.Reduce(trap, +, root = 0, comm)

    if rank == 0
        return trap
    end
end


function main()
    # integral parameters
    a = 0.0  # left endpoint
    b = pi  # right endpoint
    n = 100000000  # number of trapezoids

    integral = trapezoid(a, b, n)
    if rank == 0
        println(integral)
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
