#!/usr/bin/env python
# File: histogram.py
# Name: D.Saravanan
# Date: 25/03/2023

""" Script to compute histogram """

from random import random
from numpy import zeros

if __name__ == "__main__":
    TRIALS = 100
    print(f"Number of trails = {TRIALS}")

    SIDES = 6
    histogram = zeros(SIDES, int)

    for _ in range(TRIALS):
        r = int(random() * SIDES)
        histogram[r] = histogram[r] + 1

    print(histogram)
