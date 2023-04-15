#!/usr/bin/env julia
# File: primesumsquares.jl
# Name: D.Saravanan
# Date: 13/04/2023

""" Script to check an odd prime number has um of two squares """

function sumsquares(nval::Int64)::Int64
    """An odd prime number p, the sum of
    two squares if and only if it leaves
    the remainder 1 on division by 4."""
    if nval % 4 == 1
        println(nval)
    end
    return 0
end


function prime(number::Int64)::Int64
    """function to check prime"""
    sqrt_number::Float64 = sqrt(number)
    for i in range(2, floor(Int64, sqrt_number))
        if number % i == 0
            return 0
        end
    end
    return sumsquares(number)
end


function main()
    N::Int64 = 100
    for n in range(2, N)
        prime(n)
    end
end

main()
