#!/usr/bin/env julia
# File: binary.jl
# Name: D.Saravanan
# Date: 19/08/2021

""" Program to convert number from base 10 to base 2 """


function binary(nval::Int)::String
    """function computes binary"""
    rval::String = ""
    qval::Int = fld(nval, 2)

    while qval != 0
        rval = rval * string(nval % 2)
        qval = fld(nval, 2)
        nval = qval
    end

    return rval[end:-1:1]
end


function main()
    print("Enter number (base 10): ")
    num = parse(Int64, readline())
    println("The binary number of $num is ", binary(num))
    println("The binary number of $num is ", bitstring(num))
    println("The binary number of $num is ", string(num, base = 2))
end

main()
