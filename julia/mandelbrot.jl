#!/usr/bin/env julia
# File: mandelbrot.jl
# Name: D.Saravanan
# Date: 17/03/2023

""" Script for the Mandelbrot set """

import PyPlot as plt

plt.matplotlib.style.use("classic")
plt.rc("text", usetex = "True")
plt.rc("pgf", texsystem = "pdflatex")
plt.rc("font", family = "serif", weight = "normal", size = 10)
plt.rc("axes", labelsize = 12, titlesize = 12)
plt.rc("figure", titlesize = 12)


function mandelbrot(rmin, rmax, imin, imax)
    """an algorithm to generate an image of the Mandelbrot set"""

    max_iters = 256
    upper_bound = 2.5
    width = height = 512

    real_vals = range(rmin, rmax, width)
    imag_vals = range(imin, imax, height)

    # we will represent members as 1, non-members as 0
    mandelbrot_graph = ones(height, width)

    for x = 1:1:width
        for y = 1:1:height
            c = complex(real_vals[x], imag_vals[y])
            z = complex(0)

            for _ = 1:1:max_iters
                z = z^2 + c

                if abs(z) > upper_bound
                    mandelbrot_graph[y, x] = 0
                    break
                end
            end
        end
    end

    return mandelbrot_graph
end


function main()
    mandel = mandelbrot(-2, 2, -2, 2)

    fig, ax = plt.subplots()
    ax.imshow(mandel, extent = (-2, 2, -2, 2))
    plt.savefig("mandelbrot.png", dpi = fig.dpi)
end

main()
