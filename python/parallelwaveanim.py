#!/usr/bin/env python
# File: parallelwaveanim.py
# Name: D.Saravanan
# Date: 25/10/2022

""" Script to compute sine wave with mpi4py """

from mpi4py import MPI
import numpy as np
import matplotlib.animation as animation
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
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == -1:
        return r"$-\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N == -2:
        return r"$-\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    N = 1000

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
    if rank > 0:
        rwave[sload:eload] = wave
        comm.Send(rwave[sload:eload], dest=0)
    else:
        rwave[sload:eload] = wave
        for n in range(1, size):
            comm.Recv(rwave[n * eload :], source=n)

        fwriter = animation.writers["ffmpeg"]
        data = dict(title="Sine wave animation")
        writer = fwriter(fps=15, metadata=data)

        fig, ax = plt.subplots()
        (line1,) = ax.plot(xval, rwave, "r", lw=1, label=r"$sin(x)$")
        ax.grid(True, which="both"); ax.legend(loc="best")
        ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(formatfunc))
        ax.tick_params(which="major", direction="inout")
        ax.set(xlim=(-2 * np.pi, 2 * np.pi), ylim=(-1.5, 1.5))
        ax.set(xlabel="$x$", ylabel="$f(x)$")
        ax.set_title(r"Sine wave computed with mpi4py")

        with writer.saving(fig, "parallelwaveanim.mp4", dpi=300):

            for n in range(N):
                line1.set_data(xval[0:n], rwave[0:n])
                writer.grab_frame()


if __name__ == "__main__":
    main()
