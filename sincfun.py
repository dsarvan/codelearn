#!/usr/bin/env python3
# File: sincfun.py
# Name: D.Saravanan
# Date: 24/05/2021

""" Script to plot normalized sinc function """

import numpy as np
import matplotlib
from matplotlib import ticker
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def sinc(x: float) -> float:
    """ sinc function """
    return np.sin(np.pi * x)/(np.pi * x)

x = np.arange(-2*np.pi, 2*np.pi, 0.001)
#x, y = [n, sinc(n) for n in np.arange(-2*np.pi, 2*np.pi, 0.001)]
y = [sinc(n) for n in x]

ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

plt.plot(x, y, label=r'$\frac{\sin(\pi x)}{\pi x}$')
plt.title('sinc function', fontsize=10.)
plt.xlabel('x'); plt.ylabel(r'$\sin(\pi x)/\pi x$')
plt.grid(True, which='both'); plt.legend()
plt.savefig('sincfun.png')
