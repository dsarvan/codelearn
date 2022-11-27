#!/usr/bin/env python
# File: vectorizematrix.py
# Name: D.Saravanan
# Date: 27/11/2022

""" Vectorize matrix calculations """

import numpy as np

n = 256
a = np.random.random((n, n))
b = np.random.random((n, n))
c = np.zeros((n, n))

c = c + a * b
print(c)

from numpy.random import rand

m = 256
x = rand(n, n)
y = rand(n, n)
z = np.zeros((n, n))

z = z + x * y
print(z)
