#!/usr/bin/env python3
# File: lorenzanim.py
# Name: D.Saravanan
# Date: 19/03/2020

""" Script for lorenz system """

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

plt.style.use("classic")
plt.rcParams["text.usetex"] = True
plt.rcParams["pgf.texsystem"] = "pdflatex"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "figure.titlesize": 12,
    }
)

N = 500
x, y, z = [np.ones(N) for _ in range(3)]
#x, y, z = [np.linspace(-2*np.pi, 2*np.pi, N, endpoint=True) for _ in range(3)]
#x = np.linspace(-2*np.pi, 2*np.pi, N, endpoint=True)
#y = np.linspace(-2*np.pi, 2*np.pi, N, endpoint=True)
#z = np.linspace(-2*np.pi, 2*np.pi, N, endpoint=True)

delt = 0.01
sigma, beta, rho = 10.0, 8.0 / 3.0, 28.0

fwriter = animation.writers["ffmpeg"]
data = dict(title="Lorentz system animation")
writer = fwriter(fps=15, metadata=data)

fig, ax = plt.subplots()
(line1,) = ax.plot(x, y, color="#363737")
ax.set(xlim=(-40, 40), ylim=(-40, 40))
ax.axis('off')

with writer.saving(fig, "lorenz.mp4", dpi=200):

	for n in range(N-1):

		x[n + 1] = delt * (sigma * (y[n] - x[n])) + x[n]
		y[n + 1] = delt * (x[n] * (rho - z[n]) - y[n]) + y[n]
		z[n + 1] = delt * (x[n] * y[n] - beta * z[n]) + z[n] 
		print(x[n + 1], y[n + 1], z[n + 1])

		line1.set_data(x, y)
		writer.grab_frame()
