#!/usr/bin/env julia
# File: binary.jl
# Name: D.Saravanan
# Date: 19/08/2021

""" Program to convert number from base 10 to base 2 """

print("Enter number (base 10): ")

# reads the string
num = readline()

# parsing the string to integer
num = parse(Int64, num)

println(bitstring(num))
