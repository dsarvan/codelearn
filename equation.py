#!/usr/bin/env python3
# File: equation.py
# Name: D.Saravanan
# Date: 10/10/2020

""" Script to find solutions to equation x**2 - 7 y**2 = 1 """

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-dark")

solutions = []

s1, s2 = -10000, 10000

for x in np.arange(s1, s2, 1):
    for y in np.arange(s1, s2, 1):
        if (x ** 2) - 7 * (y ** 2) == 1:
            solutions.append((x, y))

print(solutions)

# xval = [x[0] for x in solutions]
# yval = [x[1] for x in solutions]

xval, yval = zip(*solutions)

plt.plot(xval, yval, ":")
plt.plot(xval, yval, "ob")
plt.grid(True)
plt.savefig("equation.png")
