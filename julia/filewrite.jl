#!/usr/bin/env julia
# File: filewrite.jl
# Name: D.Saravanan
# Date: 25/09/2021

""" Program to write data to a file """

file = open("institute.txt", "w")

data1 = "Institute of Mathematical Sciences\n"
write(file, data1)

data2 = "Indian Institute of Science Education and Research\n"
write(file, data2)

close(file)
