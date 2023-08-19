#!/usr/bin/env python
# File: doublfib.py
# Name: D.Saravanan
# Date: 17/11/2022

""" Script to compute nth fibonacci using doubling method """


def fibonacci(n: int) -> int:
    """compute nth fibonacci"""

    nval: int = len(bin(n)[2:])

    fib: list[int] = [0, 1, 1, 2]

    for m in range(nval - 1, -1, -1):
        fib[2] = fib[0] * (2 * fib[1] - fib[0])
        fib[3] = fib[0] * fib[0] + fib[1] * fib[1]

        if (n >> m) & 1:
            fib[0], fib[1] = fib[3], fib[2] + fib[3]
        else:
            fib[0], fib[1] = fib[2], fib[3]

    return fib[0]


if __name__ == "__main__":
    N: int = 10000
    print(fibonacci(N))
