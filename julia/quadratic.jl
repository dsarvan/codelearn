#!/usr/bin/env julia
# File: quadratic.jl
# Name: D.Saravanan
# Date: 15/08/2021

""" Program to solve for the roots of a quadratic equation """

using Printf

function roots(a::Float64, b::Float64, c::Float64)

    # calculate discriminant
    discriminant = b^2 - 4 * a * c

    # solve for the roots, depending upon the value of the discriminant
    if (discriminant > 0)
        x1 = (-b + sqrt(discriminant)) / (2 * a)
        x2 = (-b - sqrt(discriminant)) / (2 * a)
        @printf("This equation has two real roots: x1 = %.2f and x2 = %.2f\n", x1, x2)

    elseif (discriminant == 0)
        x1 = (-b) / (2 * a)
        @printf("This equation has two identical real roots: x1 = x2 = %.2f\n", x1)

    else
        real_part = (-b) / (2 * a)
        imag_part = sqrt(abs(discriminant)) / (2 * a)
        @printf(
            "This equation has complex roots: x1 = %.2f + i%.2f and x2 = %.2f - i\
    %.2f\n",
            real_part,
            imag_part,
            real_part,
            imag_part
        )

    end

end

# prompt the user for the coefficients of the equation
println("This program solves for the roots of a quadratic equation")

print("Enter the coefficients a, b, and c: ")
a, b, c = [parse(Float64, x) for x in split(readline())]

# print the coefficients
@printf("The coefficients a, b, and c are: %.1f, %.1f, %.1f\n", a, b, c)

roots(a, b, c)
