#!/usr/bin/env python
# File: meshplot.py
# Name: D.Saravanan
# Date: 18/05/2022

""" Script to plot a surface mesh given by x, y, z positions of its node points """

import numpy as np
from mayavi import mlab

phi, theta = np.mgrid[0 : np.pi : 11j, 0 : 2 * np.pi : 11j]
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)
mlab.figure(bgcolor=(1, 1, 1), fgcolor=None)
mlab.mesh(x, y, z)
mlab.mesh(x, y, z, representation="wireframe", color=(0, 0, 0))
# mlab.show()
mlab.savefig("meshgrid.png")
