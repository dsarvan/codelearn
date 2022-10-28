#!/usr/bin/env julia
# File: matvecprod.jl
# Name: D.Saravanan
# Date: 25/09/2021

""" Program to compute Matrix-Vector product """

A = [0.90 0.07 0.02 0.01; 0.00 0.93 0.05 0.02; 0.00 0.00 0.85 0.15; 0.00 0.00 0.00 1.00]
x = [0.85, 0.10, 0.05, 0.00]

show(stdout, "text/plain", A * x)
println()
