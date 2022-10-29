#!/usr/bin/env python3
# File: fermat.py
# Name: D.Saravanan
# Date: 10/10/2020
# Script for Fermat's Last Theorem

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')

N = 100
result = []

for n in range(N):
    for a in range(1,N):
        for b in range(1,N):
            for c in range(1,N):
                if (a**n) / (b**n) == (c**n):
                    result.append((a, b, c, n))

print(result)

#xval = [x[0] for x in result]
#yval = [x[1] for x in result]
#zval = [x[2] for x in result]
#
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot3D(xval, yval, zval, 'grey')
#plt.show()
