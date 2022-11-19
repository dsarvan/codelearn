#!/usr/bin/env python
# File: parallelwave.py
# Name: D.Saravanan
# Date: 25/10/2022

""" Script to compute sine wave with mpi4py """

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("classic")
plt.rc("text", usetex="True")
plt.rc("pgf", texsystem="pdflatex")
plt.rc("font", family="serif", weight="normal", size=10)
plt.rc("axes", labelsize=12, titlesize=12)
plt.rc("figure", titlesize=12)


def formatfunc(value, tick_number):
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return r"$0$"
    elif N == 1:
        return r"$\pi/2$"
    elif N == -1:
        return r"$-\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N == -2:
        return r"$-\pi$"
    elif N % 2 > 0 or N % 2 < 0:
        return r"${0}\pi/2$".format(N)
    else:
        N = N // 2
        return r"${0}\pi$".format(N)


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = 10000000

    xval = np.linspace(-2 * np.pi, 2 * np.pi, N)
    workloads = [len(xval) // size for _ in range(size)]

    for n in range(len(xval) % size):
        workloads[n] += 1

    sload = 0
    for n in range(rank):
        sload += workloads[n]
    eload = sload + workloads[rank]

    wave = np.sin(xval[sload:eload])

    rwave = np.zeros(len(xval), dtype=np.float64)
    comm.Gatherv(wave, rwave, 0)

    if rank == 0:
        fig, ax = plt.subplots()
        ax.plot(xval, rwave, "r", lw=1, label=r"$sin(x)$")
        ax.grid(True, which="both")
        ax.legend(loc="best")
        ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(formatfunc))
        ax.tick_params(which="major", direction="inout")
        ax.set(xlim=(-2 * np.pi, 2 * np.pi), ylim=(-1.5, 1.5))
        ax.set(xlabel="$x$", ylabel="$f(x)$")
        ax.set_title(r"Sine wave computed with mpi4py")
        plt.savefig("parallelwave.png", dpi=100)


if __name__ == "__main__":
    main()
