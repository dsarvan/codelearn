#!/usr/bin/env julia
# File: perfectsquare.jl
# Name: D.Saravanan
# Date: 20/10/2022

""" Test whether a number is a perfect square """

function is_square(num)
    root = Int(trunc(num^0.5))
    num == root * root
end

println(is_square(49))
println(is_square(50))
