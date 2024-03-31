#!/usr/bin/env bash


score=0 # will be used to store the score of the game

# variables below will be used to randomly get the extractable cells/fields from mine
a="1 10 -10 -1"
b="-1 0 1"
c="0 1"
d="-1 0 1 -2 -3"
e="1 2 20 21 10 0 -10 -20 -23 -2 -1"
f="1 2 3 35 30 20 22 10 0 -10 -20 -25 -30 -35 -3 -2 -1"
g="1 4 6 9 10 15 20 25 30 -30 -24 -11 -10 -9 -8 -7"

# declarations
declare -a room # declare an array room, it will represent each cell/field of our mine

printf '\n\n'
printf '%s' "     a   b   c   d   e   f   g   h   i   j"
printf '\n   %s\n' "-----------------------------------------"
