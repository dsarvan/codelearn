#!/usr/bin/env python
# File: cachefib.py
# Name: D.Saravanan
# Date: 20/03/2023

""" Script to compute nth Fibonacci number """


def fibonacci(n: int, cache) -> int:
    """compute nth fibonacci"""

    if n in cache:
        return cache[n]

    cache[n] = fibonacci(n - 1, cache) + fibonacci(n - 2, cache)
    return cache[n]


if __name__ == "__main__":
    cval: dict[int, int] = {1: 1, 2: 1}
    nval: int = 1000
    fnum: int = fibonacci(nval, cval)
    print(f"The {nval}th Fibonacci number is {fnum}")
