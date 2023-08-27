#!/usr/bin/env python
# File: dynamfib.py
# Name: D.Saravanan
# Date: 29/10/2022

""" Script to compute nth fibonacci using dynamic programming """


def fibonacci(n: int) -> int:
    """compute nth fibonacci"""

    fib: list[int] = [0, 1]

    for _ in range(n):
        fib[0] = fib[0] + fib[1]
        fib[0], fib[1] = fib[1], fib[0]

    return fib[0]


if __name__ == "__main__":
    N: int = 10000
    print(fibonacci(N))
