#!/usr/bin/env bash

is_null_field() {
    local e=$1
    if [[ -z "${room[$e]}" ]]; then
        room[$r]="."
    fi
}


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
printf '%s' "     a    b    c    d    e    f    g    h    i    j"
printf '\n   %s\n' "---------------------------------------------------"

r=0 # counter variable to keep track of how many horizontal rows have been populated
for row in $(seq 0 9); do
    printf '%d  ' "$row" # print the row numbers from 0-9

    for col in $(seq 0 9); do
        ((r+=1)) # increment the counter as we move forward in column sequence
        is_null_field $r
        printf '%s \e[33m%s\e[0m  ' "|" "${room[$r]}"
    done

    printf '%s\n' "|"
    printf '   %s\n' "---------------------------------------------------"
done
printf '\n\n'