#!/usr/bin/env julia
# File: guess.jl
# Name: D.Saravanan
# Date: 30/08/2020

""" Program for number guessing game """

option = "yes"

while option == "yes"

    N = rand(1:1:100)

    print("\nI 'm thinking of a number! Try to guess the number I 'm thinking of: ")
    num = parse(Int, readline())

    while num != N

        if num < N
            print("Too low! Guess again: ")
            num = parse(Int, readline())
        elseif num > N
            print("Too high! Guess again: ")
            num = parse(Int, readline())
        end
    end

    if num == N
        print("That's it! Would you like to play again? (yes/no) ")
        global option = readline()
    end
end

println("Thanks for playing!")
