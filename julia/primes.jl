#!/usr/bin/env julia
# File: primes.jl
# Name: D.Saravanan
# Date: 16/12/2020

""" Program to implement Sieve of Eratosthenes """

function eratosthenes(n::Int)
    isprime = trues(n)
    isprime[1] = false
    for i = 2:isqrt(n)
        if isprime[i]
            for j = i^2:i:n
                isprime[j] = false
            end
        end
    end
    return filter(x -> isprime[x], 1:n)
end

println(eratosthenes(100))
@time length(eratosthenes(10^6))
