#!/usr/bin/env python3
# File: quadratic.py
# Name: D.Saravanan
# Date: 15/08/2021

""" Script to solve for the roots of a quadratic equation 
of the form a*x**2 + b*x + c = 0 """

import math


def roots(a: int, b: int, c: int):
    """function returns root values based on discriminent condition"""

    # calculate discriminant
    discriminant: float = b ** 2 - 4 * a * c

    # solve for the roots, depending upon the value of the discriminant
    if discriminant > 0:
        x1 = (-b + math.sqrt(discriminant)) / (2 * a)
        x2 = (-b - math.sqrt(discriminant)) / (2 * a)
        print("This equation has two real roots: x1 = ", x1, " and x2 = ", x2)

    elif discriminant == 0:
        x1 = (-b) / (2 * a)
        print("This equation has two identical real roots: x1 = x2 = ", x1)

    else:
        real_part = (-b) / (2 * a)
        imag_part = math.sqrt(abs(discriminant)) / (2 * a)
        print("This equation has complex roots: x1 = ", real_part, " + i ", imag_part,
              " and x2 = ", real_part, " - i ", imag_part)


if __name__ == "__main__":

    roots(1, 2, 2)
