#!/usr/bin/env python
# File: diffnoise.py
# Name: D.Saravanan
# Date: 17/10/2022

""" Numerical differentiation with noise """

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("classic")
plt.rc("text", usetex="True")
plt.rc("pgf", texsystem="pdflatex")
plt.rc("font", family="serif", weight="normal", size=10)
plt.rc("axes", labelsize=12, titlesize=12)
plt.rc("figure", titlesize=12)

OMEGA = 100
EPSILON = 0.01

x = np.linspace(0, 2 * np.pi, 1000)

f1 = np.cos(x)
f2 = np.cos(x) + EPSILON * np.sin(OMEGA * x)

df1 = -np.sin(x)
df2 = -np.sin(x) + EPSILON * OMEGA * np.cos(OMEGA * x)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(x, f2, "r", label=r"$cos(x) + \epsilon * sin(\omega * x)$")
ax1.plot(x, f1, "b", label=r"$cos(x)$")
ax1.set(xlabel="$x$", ylabel="$f(x)$")
ax1.set(xlim=(0, 2 * np.pi), ylim=(-2.0, 2.0))
ax1.legend(loc="best")
ax1.grid(True, which="both")
ax1.set_title(r"cosine wave corrupted by a small sine wave")
ax2.plot(x, df2, "r", label=r"$sin(x) + \epsilon * \omega * cos(\omega * x)$")
ax2.plot(x, df1, "b", label=r"$sin(x)$")
ax2.set(xlabel="$x$", ylabel="$df(x)$")
ax2.set(xlim=(0, 2 * np.pi), ylim=(-2.0, 2.0))
ax2.legend(loc="best")
ax2.grid(True, which="both")
ax2.set_title(r"Numerical differentiation with noise ($\omega = 100$, $\epsilon = 0.01$)")
fig.tight_layout()
plt.savefig("diffnoise.png", dpi=100, bbox_inches="tight", pad_inches=0.1)
