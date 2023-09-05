#!/usr/bin/env julia
# File: dynamfibn.jl
# Name: D.Saravanan
# Date: 29/10/2022

""" Script to compute nth fibonacci using dynamic programming """


function fibonacci(n::Int)::BigInt
    """compute nth fibonacci"""

    fib::Vector{BigInt} = [0, 1, 1]

    for _ = 2:n
        fib[3] = fib[1] + fib[2]
        fib[1], fib[2] = fib[2], fib[3]
    end

    return n == 0 ? fib[1] : n == 1 ? fib[2] : fib[3]
end


function main()
    N::Int = 10000
    println(fibonacci(N))
end

main()
