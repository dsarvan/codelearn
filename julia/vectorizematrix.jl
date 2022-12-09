#!/usr/bin/env julia
# File: vectorizematrix.jl
# Name: D.Saravanan
# Date: 20/10/2022

""" Vectorize matrix operations """

using Random

n = 256
a = rand(n, n)
b = rand(n, n)
c = zeros(n, n)

c = c + a * b
println(c)
