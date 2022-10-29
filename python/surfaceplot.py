#!/usr/bin/env python
# File: surfaceplot.py
# Name: D.Saravanan
# Date: 26/08/2022

""" Script for surface plotting """

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use("classic")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "figure.titlesize": 10,
    }
)
plt.rcParams["text.usetex"] = True

delta = 0.1
x = np.arange(-3., 3., delta)
y = np.arange(-3., 3., delta)

xval, yval = np.meshgrid(x, y)
zval = np.sin(xval) * np.cos(yval)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(xval, yval, zval)
ax.plot_wireframe(xval, yval, zval, color="r")
ax.set(xlabel="x", ylabel="y", zlabel="z")
ax.set_title(r"Surface plot")
plt.savefig("surfaceplot.png")
