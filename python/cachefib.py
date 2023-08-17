#!/usr/bin/env python
# File: cachefib.py
# Name: D.Saravanan
# Date: 20/03/2023

""" Script to compute nth Fibonacci number """


def fibonacci(n: int) -> int:
    """compute nth fibonacci"""

    cache: dict[int, int] = {}

    if n < 3:
        return 1

    if n in cache:
        return cache[n]

    cache[n] = fibonacci(n - 1) + fibonacci(n - 2)
    return cache[n]


if __name__ == "__main__":
    fnum: int = fibonacci(30)
    print(f"The 30th Fibonacci number is {fnum}")
