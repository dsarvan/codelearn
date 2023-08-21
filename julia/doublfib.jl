#!/usr/bin/env julia
# File: doublfib.jl
# Name: D.Saravanan
# Date: 18/11/2022

""" Script to compute nth fibonacci using doubling method """


function fibonacci(n::Int)::BigInt
    """compute nth fibonacci"""

    nval::Int = length(string(n, base = 2))

    fib::Vector{BigInt} = [0, 1, 1, 2]

    for m = nval-1:-1:0
        fib[3] = fib[1] * (2 * fib[2] - fib[1])
        fib[4] = fib[1] * fib[1] + fib[2] * fib[2]

        if (n >> m) & 1 == 1
            fib[1], fib[2] = fib[4], fib[3] + fib[4]
        else
            fib[1], fib[2] = fib[3], fib[4]
        end
    end

    return fib[1]
end

function main()
    N::Int = 10000
    println(fibonacci(N))
end

main()
