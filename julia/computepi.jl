#!/usr/bin/env julia
# File: computepi.jl
# Name: D.Saravanan
# Date: 06/03/2023

""" Script to compute pi """

function computepi(N)
    """ compute pi """
    dx = 1 / N # step size
    x = i -> (i + 0.5) * dx
    pi = sum(map(i -> 4.0 / (1 + x(i)^2) * dx, 0:N))
    return pi
end

function main()
    ns = 100000000 # number of steps
    pi = computepi(ns)
    println(pi)
end

main()
