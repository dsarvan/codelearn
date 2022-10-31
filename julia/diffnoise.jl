#!/usr/bin/env julia
# File: diffnoise.jl
# Name: D.Saravanan
# Date: 17/10/2022

""" Numerical differentiation with noise """

import PyPlot as plt

plt.matplotlib.style.use("classic")
plt.rc("text", usetex = "True")
plt.rc("pgf", texsystem = "pdflatex")
plt.rc("font", family = "serif", weight = "normal", size = 10)
plt.rc("axes", labelsize = 12, titlesize = 12)
plt.rc("figure", titlesize = 12)

OMEGA = 100
EPSILON = 0.01

x = LinRange(0, 2 * pi, 1000)

f1 = cos.(x)
f2 = cos.(x) + EPSILON * sin.(OMEGA .* x)

df1 = -sin.(x)
df2 = -sin.(x) + EPSILON * OMEGA * cos.(OMEGA .* x)

fig, ax = plt.subplots(2, 1)
ax[1].plot(x, f2, "r", label = raw"$cos(x) + \epsilon * sin(\omega * x)$")
ax[1].plot(x, f1, "b", label = raw"$cos(x)$")
ax[1].set(xlabel = raw"$x$", ylabel = raw"$f(x)$")
ax[1].set(xlim = (0, 2 * pi), ylim = (-2.0, 2.0), yticks = -2.0:0.5:2.0)
ax[1].tick_params(direction = "in")
ax[1].legend(loc = "best")
ax[1].grid(true, which = "both")
ax[1].set_title(raw"cosine wave corrupted by a small sine wave")
ax[2].plot(x, df2, "r", label = raw"$sin(x) + \epsilon * \omega * cos(\omega * x)$")
ax[2].plot(x, df1, "b", label = raw"$sin(x)$")
ax[2].set(xlabel = raw"$x$", ylabel = raw"$df(x)$")
ax[2].set(xlim = (0, 2 * pi), ylim = (-2.0, 2.0), yticks = -2.0:0.5:2.0)
ax[2].tick_params(direction = "in")
ax[2].legend(loc = "best")
ax[2].grid(true, which = "both")
ax[2].set_title(raw"Numerical differentiation with noise ($\omega = 100$, $\epsilon = 0.01$)")
fig.tight_layout()
plt.savefig("diffnoise.png", dpi = 100, bbox_inches = "tight", pad_inches = 0.1)
