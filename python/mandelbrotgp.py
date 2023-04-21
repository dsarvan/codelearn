#!/usr/bin/env python
# File: mandelbrotgp.py
# Name: D.Saravanan
# Date: 21/04/2023

""" Script for the Mandelbrot set with gnuplot """

import os
import numpy as np


def mandelbrot(rmin, rmax, imin, imax):
    """an algorithm to generate an image of the Mandelbrot set"""

    max_iters = 256
    upper_bound = 2.5
    width = height = 512

    real_vals = np.linspace(rmin, rmax, width)
    imag_vals = np.linspace(imin, imax, height)

    # we will represent members as 1, non-members as 0.
    mandelbrot_graph = np.ones((height, width), dtype=np.int32)

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

	mandel = mandelbrot(-2, 2, -2, 2)

	gp = os.popen("gnuplot -persist", "w")
	gp.write("set colorsequence classic\n")
	gp.write("set output 'mandelbrotgp.png'\n")
	gp.write("set terminal pngcairo font 'Times,12'\n")
	gp.write("set autoscale xfix; set autoscale yfix\n")
	gp.write("set cbrange [0:1]; set autoscale cbfix\n")
	gp.write("set palette defined (0 'blue', 1 'white')\n")
	gp.write("plot '-' matrix with image pixels notitle\n")
	for i in range(512):
		for j in range(512):
			gp.write("%d " %mandel[i][j])
		gp.write("\n")
	gp.write("e\n")
	gp.close()
