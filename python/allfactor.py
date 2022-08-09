#!/usr/bin/env python3
# File: allfactor.py
# Name: D.Saravanan
# Date: 23/09/2020

""" Script to find all factors of an integer """


def factors(nval: int) -> list:
    """factors of an integer"""
    return [n for n in range(1, nval + 1) if nval % n == 0]


if __name__ == "__main__":

    NUM = float(input("Enter number: "))

    if NUM > 0 and NUM.is_integer():
        NUM = int(NUM)
        print(f"The factors of {NUM} are {factors(NUM)}")
    else:
        print("INUMid input, enter a positive integer.")
