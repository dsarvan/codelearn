#!/usr/bin/env python
# File: primes.py
# Name: D.Saravanan
# Date: 26/08/2022

""" Script that checks whether a number is prime """

import math


def prime(number: int) -> str:
    """function to check prime"""
    sqrt_number: float = math.sqrt(number)
    for i in range(2, int(sqrt_number) + 1):
        if number % i == 0:
            return f"{number} is not a prime number"
    return f"{number} is a prime number"


if __name__ == "__main__":
    for n in range(2, 31):
        print(prime(n))
