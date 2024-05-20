#!/usr/bin/env python
# File: coordinate.py
# Name: D.Saravanan
# Date: 17/05/2024

""" Script to display coordinate system using vpython """

from vpython import arrow, color, label, points, scene, vec, vector

height, width, depth = 10.0, 10.0, 10.0

scene.title = "<h2>Coordinate system</h2>"
scene.width, scene.height = 600, 600
scene.background = color.white
scene.center = vector(0, 0, 0)
scene.range = 1.5 * width

x0, y0, z0 = vector(-width, 0, 0), vector(0, -height, 0), vector(0, 0, -depth)

arrow(pos=x0, axis=vector(2 * width, 0, 0), shaftwidth=0.15, color=color.black)
arrow(pos=y0, axis=vector(0, 2 * height, 0), shaftwidth=0.15, color=color.black)
arrow(pos=z0, axis=vector(0, 0, 2 * depth), shaftwidth=0.15, color=color.black)

label(pos=vec(width, -1, 0), text="x", height=30, box=False, opacity=0)
label(pos=vec(-1, height, 0), text="y", height=30, box=False, opacity=0)
label(pos=vec(-1, 0, depth), text="z", height=30, box=False, opacity=0)

points(pos=vector(5, 5, 0))

scene.caption = "\nPress right mouse button and drag"
