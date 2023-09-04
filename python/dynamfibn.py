#!/usr/bin/env python
# File: dynamfibn.py
# Name: D.Saravanan
# Date: 29/10/2022

""" Script to compute nth fibonacci using dynamic programming """


def fibonacci(n: int) -> int:
    """compute nth fibonacci"""

    fib: list[int] = [0, 1, 1]

    for _ in range(1, n):
        fib[2] = fib[0] + fib[1]
        fib[0], fib[1] = fib[1], fib[2]

    return fib[0] if n == 0 else fib[1] if n == 1 else fib[2]


if __name__ == "__main__":
    N: int = 10
    print(fibonacci(N))
