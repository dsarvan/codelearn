#!/usr/bin/env julia
# File: scrabble.jl
# Name: D.Saravanan
# Date: 20/10/2022

""" Analyze words for a given set of files """

using Printf

function getWordList(WordList::Set{String})
    inn = open("scrabble.txt", 'r')

    while (line = readline(inn)) != ""
        words = split(line)
        for w in words
            push!(WordList, uppercase(w))
        end
    end
    close(inn)
end # getWordList

function main()
    WordList = Set{String}()
    getWordList(WordList)   # Words in the dictionary

    Tiles = Dict(
        'A' => 1,
        'B' => 3,
        'C' => 3,
        'D' => 2,
        'E' => 1,
        'F' => 4,
        'G' => 2,
        'H' => 4,
        'I' => 1,
    )
    Hand = ['A', 'B', 'D', 'E', 'F', 'G', 'H']  # array of Char
    hsize = length(Hand)
    println("Letters in Hand: $Hand")
end # main

main()
