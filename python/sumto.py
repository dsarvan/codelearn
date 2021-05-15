#!/usr/bin/env python3
# File: sumto.py
# Name: D.Saravanan
# Date: 14/05/2021
# Script to use yield and static type

def numbers():
    """ lazy function, it only
    creates numbers as requested """
    for num in range(1024):
        yield num

def sumto(number: int) -> int:
    """ accept an integer value for the 
    n parameter and return an integer result """
    summ: int = 0
    for num in numbers():
        if num == number: break
        summ += num
    return summ

print(sumto(15))
