#!/usr/bin/env python
# File: allfactor.py
# Name: D.Saravanan
# Date: 23/09/2020

""" Script to find all factors of a positive integer """


def factors(nval: int) -> list[int]:
    """factors of an integer"""
    return [n for n in range(1, nval + 1) if nval % n == 0]


if __name__ == "__main__":
    while True:
        NUM: int = int(input("Enter a positive integer: "))

        if NUM >> 31 == 0:
            print(f"The factors of {NUM} are {factors(NUM)}")
            break

        print("Invalid input, enter a positive integer.")
