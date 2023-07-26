#!/usr/bin/env python
# File: factorial.py
# Name: D.Saravanan
# Date: 26/08/2022

""" Script to compute factorial """

import math

import numpy as np


def factorial1(n: int) -> int:
    """not suggested (N = 22)"""
    f = np.ones(n + 1)
    for i in range(1, n + 1):
        f[i] = i * f[i - 1]
    return int(f[-1])


def factorial2(n: int) -> int:
    """not suggested (N = 22)"""
    f = np.linspace(1, n, n)
    return int(np.prod(f))


def factorial3(n: int) -> int:
    """well suggested"""
    product = 1
    for i in range(1, n + 1):
        product = product * i
    return product


def factorial4(n: int) -> int:
    """well suggested"""
    return math.factorial(n)


def factorial5(n: int) -> int:
    """good but not better (N = 498)"""
    return n * factorial5(n - 1) if n else 1


if __name__ == "__main__":
    N: int = 22
    print(factorial1(N))
    print(factorial2(N))
    print(factorial3(N))
    print(factorial4(N))
    print(factorial5(N))
