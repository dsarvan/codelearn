#!/usr/bin/env julia
# File: dynamfib.jl
# Name: D.Saravanan
# Date: 29/10/2022

""" Script to compute nth fibonacci using dynamic programming """


function fibonacci(n::Int)::BigInt
    """compute nth fibonacci"""

    fib::Vector{BigInt} = [0, 1]

    for _ = 1:n
        fib[1] = fib[1] + fib[2]
        fib[1], fib[2] = fib[2], fib[1]
    end

    return fib[1]
end


function main()
    N::Int = 10000
    println(fibonacci(N))
end

main()
