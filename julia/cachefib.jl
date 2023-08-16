#!/usr/bin/env julia
# File: cachefib.jl
# Name: D.Saravanan
# Date: 20/03/2023

""" Script to compute nth Fibonacci number """


function fibonacci(n::Int, cache)::BigInt
    """compute nth fibonacci"""

    if n in keys(cache)
        return cache[n]
    end

    cache[n] = fibonacci(n - 1, cache) + fibonacci(n - 2, cache)
    return cache[n]
end


function main()
    cache = Dict{Int,BigInt}(1 => 1, 2 => 1)
    nval::Int = 1000
    fnum::BigInt = fibonacci(nval, cache)
    println("The $(nval)th Fibonacci number is $fnum")
end

main()
