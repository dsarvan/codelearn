#!/usr/bin/env python3
# File: prime.py
# Name: D.Saravanan
# Date: 09/08/2021

""" Script that checks whether a number is prime """

import math


def prime(number: int) -> bool:
    """function to check prime"""
    sqrt_number: float = math.sqrt(number)
    for index in range(2, int(sqrt_number) + 1):
        if (number / index).is_integer():
            return False
    return True


if __name__ == "__main__":
    print(f"Check number(10,000,000) = {prime(10_000_000)}")
    print(f"Check number(10,000,019) = {prime(10_000_019)}")
