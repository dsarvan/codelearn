#!/usr/bin/env python3
# File: lorenzanim.py
# Name: D.Saravanan
# Date: 19/03/2020

""" Script for lorenz system """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

plt.style.use("classic")
plt.rc("text", usetex="True")
plt.rc("pgf", texsystem="pdflatex")
plt.rc("font", family="serif", weight="normal", size=10)
plt.rc("axes", labelsize=12, titlesize=12)
plt.rc("figure", titlesize=12)

N = 5000
x, y, z = [np.ones(N) for _ in range(3)]

DELT = 0.01
SIGMA, BETA, RHO = 10.0, 8.0 / 3.0, 28.0

fwriter = animation.writers["ffmpeg"]
data = dict(title="Lorentz system animation")
writer = fwriter(fps=15, metadata=data)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
(line1,) = ax1.plot(x, z, color="#363737")
ax1.set(xlim=(-20, 22), ylim=(0, 55))
ax1.axis("off")
(line2,) = ax2.plot(y, z, color="#363737")
ax2.set(xlim=(-30, 30), ylim=(0, 55))
ax2.axis("off")
(line3,) = ax3.plot(y, x, color="#363737")
ax3.set(xlim=(-30, 30), ylim=(-20, 22))
ax3.axis("off")


with writer.saving(fig, "lorenzanim.mp4", dpi=200):

    for n in range(N - 1):
        x[n + 1] = DELT * (SIGMA * (y[n] - x[n])) + x[n]
        y[n + 1] = DELT * (x[n] * (RHO - z[n]) - y[n]) + y[n]
        z[n + 1] = DELT * (x[n] * y[n] - BETA * z[n]) + z[n]

        line1.set_data(x[0:n], z[0:n])
        line2.set_data(y[0:n], z[0:n])
        line3.set_data(y[0:n], x[0:n])
        writer.grab_frame()
