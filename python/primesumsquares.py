#!/usr/bin/env python
# File: primesumsquares.py
# Name: D.Saravanan
# Date: 12/04/2023

""" Script to check an odd prime number has sum of two squares """

from math import sqrt


def sumsquares(nval: int) -> int:
    """An odd prime number p, the sum of
    two squares if and only if it leaves
    the remainder 1 on division by 4."""
    if nval % 4 == 1:
        print(nval)
    return 0


def prime(number: int) -> int:
    """function to check prime"""
    sqrt_number: float = sqrt(number)
    for i in range(2, int(sqrt_number) + 1):
        if number % i == 0:
            return 0
    return sumsquares(number)


if __name__ == "__main__":
    N: int = 100
    for n in range(2, N):
        prime(n)
