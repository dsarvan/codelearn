#!/usr/bin/env python3
# File: signalgen.py
# Name: D.Saravanan
# Date: 16/08/2021

""" Script to generate sinusoidal signal """

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')
matplotlib.use('TkAgg')

# sampling frequency in Hz
fs = 1000 

# generate the time vector
t = np.arange(1000)/fs
signal1 = np.sin(2*np.pi*5*t)
signal2 = np.sin(2*np.pi*7*t)
signal3 = signal1 * signal2

plt.plot(t, signal3)
plt.xlabel(r'$\theta$($\degree$)')
plt.ylabel(r'$P(\theta)$')
plt.xlim(0,1); plt.grid()
plt.savefig('signalplot.png')
