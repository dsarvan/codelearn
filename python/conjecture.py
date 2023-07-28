#!/usr/bin/env python
# File: conjecture.py
# Name: D.Saravanan
# Date: 13/07/2021

""" Script for 3n + 1 collatz conjecture """


def conjecture(n: int) -> list[int]:
    """collatz conjecture"""
    sequence: list[int] = [n]

    while n != 1:
        n = n // 2 if n % 2 == 0 else (3 * n) + 1
        sequence = sequence + [n]

    return sequence


if __name__ == "__main__":
    print(conjecture(27))
