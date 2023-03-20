#!/usr/bin/env python
# File: cachefib.py
# Name: D.Saravanan
# Date: 20/03/2023

""" Script to compute nth Fibonacci number """

cache = {}


def fibonacci(n):
    """compute fibonacci"""

    if n < 3:
        return 1

    if n in cache:
        return cache[n]

    cache[n] = fibonacci(n - 1) + fibonacci(n - 2)
    return cache[n]


fnum = fibonacci(30)
print(f"The 30th Fibonacci number is {fnum}")
