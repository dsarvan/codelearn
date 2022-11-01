#!/usr/bin/env python3
# File: lorenzsys.py
# Name: D.Saravanan
# Date: 19/03/2020

""" Script for lorenz system """

import numpy as np
import matplotlib.pyplot as plt

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

for n in range(N - 1):
    x[n + 1] = DELT * (SIGMA * (y[n] - x[n])) + x[n]
    y[n + 1] = DELT * (x[n] * (RHO - z[n]) - y[n]) + y[n]
    z[n + 1] = DELT * (x[n] * y[n] - BETA * z[n]) + z[n]


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.plot(x, z, color="#363737")
ax1.axis("off")
ax1.tick_params(axis="both", left=False, top=False, right=False, bottom=False)
ax2.plot(y, z, color="#363737")
ax2.axis("off")
ax2.tick_params(axis="both", left=False, top=False, right=False, bottom=False)
ax3.plot(y, x, color="#363737")
ax3.axis("off")
ax3.tick_params(axis="both", left=False, top=False, right=False, bottom=False)
plt.tight_layout()
# plt.text(-110, -20, r"Lorenz attractor when $\rho = 28$, $\sigma = 10$, and $\beta = 8/3$")
plt.title(r"Lorenz attractor when $\rho = 28$, $\sigma = 10$, and $\beta = 8/3$", x=-0.55, y=-0.03)
plt.savefig("lorenzsys.png", dpi=100, bbox_inches="tight", pad_inches=0.0)
