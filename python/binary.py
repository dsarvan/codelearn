#!/usr/bin/env python3
# File: binary.py
# Name: D.Saravanan
# Date: 31/05/2021

""" Script to convert number from base 10 to base 2 """


def binary(nval: int) -> str:
    """function computes binary"""
    rval: str = ""
    qval: int = nval // 2

    while qval != 0:
        rval = rval + str(nval % 2)
        qval = nval // 2
        nval = qval

    return rval[::-1]


if __name__ == "__main__":
    num = int(input("Enter number (base 10): "))
    print(f"The binary number of {num} is {binary(num)}")
    print(f"The binary number of {num} is {bin(num)[2:]}")
