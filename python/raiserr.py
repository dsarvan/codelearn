#!/usr/bin/env python
# File: raiserr.py
# Name: D.Saravanan
# Date: 29/10/2022

""" Script to raise TypeError and ValueError """

import math

if __name__ == "__main__":
    n = input("Enter number: ")

    if isinstance(n, str):
        n = int(float(n))
    if isinstance(n, float):
        raise TypeError("number should be of type integer")
    if n <= 0:
        raise ValueError("number should be greater than zero")

    print(f"{n} square is {n*n}")
    print(f"{n} square root is {math.sqrt(n)}")
