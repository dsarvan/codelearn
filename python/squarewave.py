#!/usr/bin/env python
# File: squarewave.py
# Name: D.Saravanan
# Date: 26/08/2022

""" Script to create square wave by summing sine wave """

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
square_wave = np.sin(2*np.pi*time)

fwriter = animation.writers["ffmpeg"]
data = dict(title="Square wave animation")
writer = fwriter(fps=15, metadata=data)

fig, ax = plt.subplots()
(line1,) = ax.plot(square_wave, "r", lw=1)

with writer.saving(fig, "squarewave.mp4", 120):

    line1.set_ydata(square_wave)
    writer.grab_frame()

    for n in range(3, 10001, 2):
        square_wave += (1/n) * np.sin(2*np.pi*n*time)

        line1.set_ydata(square_wave)
        writer.grab_frame()
