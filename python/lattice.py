#!/usr/bin/env python
# File: lattice.py
# Name: D.Saravanan
# Date: 17/05/2024

""" Script to display lattice structure using vpython """

from vpython import color, scene, sphere, vector

scene.title = "<h2>Lattice structure</h2>"
scene.background = color.white
scene.width, scene.height = 600, 600

SPACING = 5

for x in range(-SPACING, SPACING):
    for y in range(-SPACING, SPACING):
        for z in range(-SPACING, SPACING):
            sphere(pos=vector(x, y, z), radius=0.5, color=color.red)


scene.caption = "\nPress right mouse button and drag"
