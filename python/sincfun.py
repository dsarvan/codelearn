#!/usr/bin/env python3
# File: sincfun.py
# Name: D.Saravanan
# Date: 24/05/2021
# Script to plot normalized sinc function

import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def sinc(value: float) -> float:
    """ sinc function """
    return math.sin(math.pi * value)/(math.pi * value)

xval, yval = [], []
for nval in range(1,101):
    xval.append(nval)
    yval.append(sinc(nval))

plt.plot(xval, yval)
plt.savefig('sincfun.png')
