#!/usr/bin/env python3
# File: signalgen.py
# Name: D.Saravanan
# Date: 16/08/2021

""" Script to generate sinusoidal signal """

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-dark")
matplotlib.use("TkAgg")

# sampling frequency in Hz
fs: int = 1000

# generate the time vector
t: float = np.arange(1000) / fs
signal1: float = np.sin(2 * np.pi * 5 * t)
signal2: float = np.sin(2 * np.pi * 7 * t)
signal3: float = signal1 * signal2

plt.plot(t, signal3)
plt.xlabel(r"$\theta^{\degree}$")
plt.ylabel(r"$P(\theta)$")
plt.xlim(0, 1); plt.grid()
plt.savefig("signalplot.png")
