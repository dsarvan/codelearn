#!/usr/bin/env python
# File: fractional.py
# Name: D.Saravanan
# Date: 16/09/2022

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

x = np.linspace(0, 10, 1000, endpoint=True)
y = x

fwriter = animation.writers["ffmpeg"]
data = dict(title="Fractional Calculus")
writer = fwriter(fps=15, metadata=data)

fig, ax = plt.subplots()
(line1,) = ax.plot(x, y, c="#B00020", lw=1)
nval_text = ax.text(0.90, 0.90, "", transform=ax.transAxes)
ax.plot(x, x**0, c="#014D4E", lw=1)
ax.plot(x, x**1, c="#A88905", lw=1)
ax.plot(x, x**2, c="#960056", lw=1)
ax.plot(x, x**3, c="#856798", lw=1)
ax.spines[["right", "top"]].set_visible(False)
ax.tick_params(axis="both", right=False, top=False)
ax.tick_params(which="major", direction="inout")
ax.set(xlim=(0, 5), ylim=(0, 10))
ax.set_title(r"Fractional Calculus", pad=20)

with writer.saving(fig, "fractional.mp4", dpi=200):

	line1.set_ydata(y)
	writer.grab_frame()

	for n in np.arange(0, 3.001, 0.001):
		y = x**n

		line1.set_ydata(y)
		nval_text.set_text(f"n = {round(n, 3)}")
		writer.grab_frame()
