#!/usr/bin/env julia
# File: sincfun.jl
# Name: D.Saravanan
# Date: 19/08/2021

""" Program to plot normalized sinc function """

using Plots

#@time function sinc(x::Float64)::Float64
#    return sin(pi*x)/(pi*x)
#end

@time x = -2*pi:0.001:2*pi
@time y = [sinc(n) for n in x]

plot(x, y, title = "sinc function")
savefig("sincfun.png")
