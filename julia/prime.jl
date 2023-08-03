#!/usr/bin/env julia
# File: prime.jl
# Name: D.Saravanan
# Date: 06/11/2023

""" Script that checks whether a number is prime """


function prime(number::Int)::Bool
    """function to check prime"""
    sqrt_number::Float64 = sqrt(number)
    for index in range(2, floor(Int, sqrt_number))
        if isinteger(number / index)
            return false
        end
    end
    return true
end


function main()
    println("Check number(10,000,000) = ", prime(10_000_000))
    println("Check number(10,000,019) = ", prime(10_000_019))
end

main()
