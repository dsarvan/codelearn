#!/usr/bin/env python
# File: mandelbrot.py
# Name: D.Saravanan
# Date: 03/04/2023

""" Script for the Mandelbrot set """

from time import time

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("classic")
plt.rc("text", usetex="True")
plt.rc("pgf", texsystem="pdflatex")
plt.rc("font", family="serif", weight="normal", size=10)
plt.rc("axes", labelsize=12, titlesize=12)
plt.rc("figure", titlesize=12)


def mandelbrot(rmin, rmax, imin, imax):
    """an algorithm to generate an image of the Mandelbrot set"""

    max_iters = 256
    upper_bound = 2.5
    width = height = 512

    real_vals = np.linspace(rmin, rmax, width)
    imag_vals = np.linspace(imin, imax, height)

    # we will represent members as 1, non-members as 0.

    mandelbrot_graph = np.ones((height, width), dtype=np.float32)

    for x in range(width):
        for y in range(height):
            c = np.complex64(real_vals[x] + imag_vals[y] * 1j)
            z = np.complex64(0)

            for _ in range(max_iters):
                z = z**2 + c

                if np.abs(z) > upper_bound:
                    mandelbrot_graph[y, x] = 0
                    break

    return mandelbrot_graph


if __name__ == "__main__":
    t1 = time()
    mandel = mandelbrot(-2, 2, -2, 2)
    t2 = time()
    mandel_time = t2 - t1

    t1 = time()
    fig = plt.figure(1)
    plt.imshow(mandel, extent=(-2, 2, -2, 2))
    plt.savefig("mandelbrot.png", dpi=fig.dpi)
    t2 = time()

    dump_time = t2 - t1

    print(f"It took {mandel_time} seconds to calculate the Mandelbrot graph.")
    print(f"It took {dump_time} seconds to dump the image.")
