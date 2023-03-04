#!/usr/bin/env python
# File: surfaceplot.py
# Name: D.Saravanan
# Date: 26/08/2022

""" Script for surface plotting """

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("classic")
plt.rc("text", usetex="True")
plt.rc("pgf", texsystem="pdflatex")
plt.rc("font", family="serif", weight="normal", size=10)
plt.rc("axes", labelsize=12, titlesize=12)
plt.rc("figure", titlesize=12)

DELTA = 0.1
x = np.arange(-3.0, 3.0, DELTA)
y = np.arange(-3.0, 3.0, DELTA)

xval, yval = np.meshgrid(x, y)
zval = np.sin(xval) * np.cos(yval)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(xval, yval, zval, cmap="viridis")
ax.plot_wireframe(xval, yval, zval, color="k")
ax.set(xlabel="x", ylabel="y", zlabel="z")
ax.set_title(r"Surface plot")
plt.savefig("surfaceplot.png")
