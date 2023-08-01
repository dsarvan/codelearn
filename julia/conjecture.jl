#!/usr/bin/env julia
# File: conjecture.jl
# Name: D.Saravanan
# Date: 06/11/2023

""" Script for 3n + 1 collatz conjecture """


function conjecture(n::Int)::Vector{Int}
    """collatz conjecture"""
    sequence = [n]

    while n != 1
        n = n % 2 == 0 ? fld(n, 2) : (3 * n) + 1
        #append!(sequence, n)
        sequence = vcat(sequence, [n])
    end

    return sequence
end


function main()
    println(conjecture(27))
end

main()
