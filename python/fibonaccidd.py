#!/usr/bin/env python
# File: fibonaccid.py
# Name: D.Saravanan
# Date: 29/10/2022

""" Solving standard fibonacci with dynamic programming """
# python -X int_max_str_digits=0 fibonaccid.py


def fibonacci(n: int) -> int:
    """compute nth fibonacci"""

    fib: list[int] = [0, 1, 1]

    for _ in range(1, n):
        fib[2] = fib[0] + fib[1]
        fib[0], fib[1] = fib[1], fib[2]

    return fib[0] if n == 0 else fib[1] if n == 1 else fib[2]


if __name__ == "__main__":
    N: int = 100000
    print("Invalid input" if N < 0 else fibonacci(N))
