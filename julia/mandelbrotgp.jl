#!/usr/bin/env julia
# File: mandelbrotgp.jl
# Name: D.Saravanan
# Date: 23/03/2023

""" Script for the Mandelbrot set with gnuplot """

using Gnuplot

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

    @gp "set colorsequence classic" :-
    @gp :- "set output 'mandelbrotgp.png'" :-
    @gp :- "set terminal pngcairo font 'Times,12'" :-
    @gp :- "set autoscale xfix; set autoscale yfix" :-
    @gp :- "set cbrange [0:1]; set autoscale cbfix" :-
    @gp :- "set palette defined (0 'blue', 1 'white')" :-
    @gp :- mandel "with image pixels notitle"
end

main()
