#!/usr/bin/env python3
# File: conjecture.py
# Name: D.Saravanan
# Date: 13/07/2021

""" Script for 3N + 1 conjecture """

N: int = 7
print(N)

while N != 1:

    if N % 2 == 0:
        N = N / 2
    else:
        N = (3 * N) + 1

    print(int(N))
