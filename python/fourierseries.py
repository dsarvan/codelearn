#!/usr/bin/env python
# File: fourierseries.py
# Name: D.Saravanan
# Date: 26/08/2022

""" Script for fourier series and fourier transforms """

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("classic")
plt.rcParams["text.usetex"] = True
plt.rcParams["pgf.texsystem"] = "pdflatex"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "figure.titlesize": 10,
    }
)

if __name__ == "__main__":

    fsamp = 5000
    t = np.arange(0, 3, 1/fsamp)

    square_wave = 1
    for n in range(1, 10001, 2):
        square_wave += (1/n) * np.sin(2*np.pi*n*t)

    sawtooth_wave = 1
    for n in range(1, 10001, 1):
        sawtooth_wave += (-1)**(n+1) * (1/n) * np.sin(2*np.pi*n*t)

    fig, ax = plt.subplots()
    ax.plot(t, square_wave, "r")
    ax.plot(t, sawtooth_wave, "b")
    ax.grid(True)
    plt.savefig("fourierseries.png")
