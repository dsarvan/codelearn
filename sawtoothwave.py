#!/usr/bin/env python
# File: sawtoothwave.py
# Name: D.Saravanan
# Date: 26/08/2022

""" Script to create sawtooth wave by summing sine wave """

import numpy as np
import matplotlib.animation as animation
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

freq = 5000
time = np.linspace(0, 2, freq)
sawtooth_wave = np.sin(2*np.pi*time)

fwriter = animation.writers["ffmpeg"]
data = dict(title="Sawtooth wave animation")
writer = fwriter(fps=15, metadata=data)

fig, ax = plt.subplots()
(line1,) = ax.plot(sawtooth_wave, "r", lw=1)

with writer.saving(fig, "sawtoothwave.mp4", 120):

    line1.set_ydata(sawtooth_wave)
    writer.grab_frame()

    for n in range(2, 10001, 1):
        sawtooth_wave += (-1)**(n-1) * (1/n) * np.sin(2*np.pi*n*time)

        line1.set_ydata(sawtooth_wave)
        writer.grab_frame()
