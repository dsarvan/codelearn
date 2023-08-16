#!/usr/bin/env python
# File: fibonaccid.py
# Name: D.Saravanan
# Date: 29/10/2022

""" Solving standard fibonacci with dynamic programming """

def fibonacci(n):
    fib = [0, 1, None]
    if n < 0:
        return "Invalid input"
    elif n == 0:
        return fib[0]
    elif n == 1:
        return fib[1]
    else:
        for _ in range(2, n+1):
            fib[-1] = fib[0] + fib[1]
            fib[0] = fib[1]
            fib[1] = fib[-1]
        return fib[-1]


if __name__ == "__main__":
    N = 10
    print(fibonacci(N))
