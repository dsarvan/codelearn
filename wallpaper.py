#!/usr/bin/env python3
# File: wallpaper.py
# Name: D.Saravanan
# Date: 17/05/2021
# Script to draw a wallpaper

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

corna, cornb, side = 1, 1, 10
ivalues, jvalues = [], []

for i in range(1, 101):
    for j in range(1, 101):
        x = corna + i * side/100
        y = cornb + j * side/100

        if (x**2 + y**2)%2 == 0:
            ivalues.append(i)
            jvalues.append(j)

plt.plot(ivalues, jvalues)
plt.show()
