#!/usr/bin/env julia
# File: parallelwave.jl
# Name: D.Saravanan
# Date: 25/10/2022

""" Script to compute sine wave with MPI """

import MPI
MPI.Init()
import PyPlot as plt
using LaTeXStrings

plt.matplotlib.style.use("classic")
plt.rc("text", usetex = "True")
plt.rc("pgf", texsystem = "pdflatex")
plt.rc("font", family = "serif", weight = "normal", size = 10)
plt.rc("axes", labelsize = 12, titlesize = 12)
plt.rc("figure", titlesize = 12)


function formatfunc(value, tick_number)
    N = Int(round(2 * value / pi))
    if N == 0
        return L"0"
    elseif N == 1
        return L"\pi/2"
    elseif N == -1
        return L"-\pi/2"
    elseif N == 2
        return L"\pi"
    elseif N == -2
        return L"-\pi"
    elseif N % 2 > 0 || N % 2 < 0
        return L"%$N\pi/2"
    else
        N = div(N, 2)
        return L"%$N\pi"
    end
end


function main()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    N = 10000000

    xval = LinRange(-2 * pi, 2 * pi, N)
    workloads = [trunc(Int, length(xval) / size) for _ in range(1, size)]


    for n in range(1, length(xval) % size)
        workloads[n] += 1
    end

    sload = 0
    for n in range(1, rank)
        sload += workloads[n+1]
    end
    eload = sload + workloads[rank+1]

    wave = sin.(xval[sload+1:eload])

    rwave = zeros(Float64, length(xval))
    if rank > 0
        MPI.Send(wave, dest = 0, comm)
    else
        rwave[sload+1:eload] = wave
        for n in range(1, size - 1)
            rwave[n*eload+1:(n+1)*eload] = MPI.Recv!(wave, source = n, comm)
        end

        fig, ax = plt.subplots()
        ax.plot(xval, rwave, "r", lw = 1, label = raw"$sin(x)$")
        ax.grid(true, which = "both")
        ax.legend(loc = "best")
        ax.xaxis.set_major_locator(plt.matplotlib.ticker.MultipleLocator(pi / 2))
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(formatfunc))
        ax.tick_params(which = "major", direction = "inout")
        ax.set(xlim = (-2 * pi, 2 * pi), ylim = (-1.5, 1.5))
        ax.set(xlabel = raw"$x$", ylabel = raw"$f(x)$")
        ax.set_title(raw"Sine wave computed with mpi4py")
        plt.savefig("parallelwave.png", dpi = 100)

    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()
