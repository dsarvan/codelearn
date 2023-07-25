#!/usr/bin/env julia
# File: allfactor.jl
# Name: D.Saravanan
# Date: 23/09/2020

""" Script to find all factors of a positive integer """

function factors(nval::Int)::Vector{Int}
    """factors of an integer"""
    return [n for n in range(1, nval) if nval % n == 0]
end

function main()

    while true
        print("Enter a positive integer: ")
        NUM::Int = parse(Int, readline())

        if NUM >> 31 == 0
            println("The factors of $NUM are ", factors(NUM))
            break
        end

        println("Invalid input, enter a positive integer.")
    end
end

main()
