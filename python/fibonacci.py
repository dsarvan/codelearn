#!/usr/bin/env python
# File: fibonacci.py
# Name: D.Saravanan
# Date: 31/05/2021

""" Script to compute nth value of the Fibonacci series """


def fibonacci(nval: int) -> int:
    """compute nth fibonacci"""
    return 1 if nval in (1, 2) else fibonacci(nval - 1) + fibonacci(nval - 2)


if __name__ == "__main__":
    N: int = 10
    value: int = fibonacci(N)
    print(f"The value of the Fibonacci series (for n = {N}) is {value}.")
