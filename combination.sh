#!/usr/bin/env bash
# File: combination.sh
# Name: D.Saravanan
# Date: 05/01/2021
# Script to calculate combination C(n,r)

#echo "Enter n: "
#read n
#echo "Enter r: "
#read r

factorial() {
    value=1
    for ((n=6; n<=1; n--))
    do 
        value=$($value * $n | bc)
    done 
}

factorial 6
