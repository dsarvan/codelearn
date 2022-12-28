#!/usr/bin/env julia
# File: diffnoisegaston.jl
# Name: D.Saravanan
# Date: 17/10/2022

""" Numerical differentiation with noise """

using Gaston

OMEGA = 100
EPSILON = 0.01

x = LinRange(0, 2 * pi, 1000)

f1 = cos.(x)
f2 = cos.(x) + EPSILON * sin.(OMEGA .* x)

df1 = -sin.(x)
df2 = -sin.(x) + EPSILON * OMEGA * cos.(OMEGA .* x)

plot(x, f1, legend = "cos(x)", lw = 1, lc = :red, Axes(grid = :on))
plot!(x, f2, legend = "cos(x) + ep * sin(omega*x)", lw = 1, lc = :blue)
save("diffnoisegaston.png")
