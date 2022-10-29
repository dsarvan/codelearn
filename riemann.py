#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

xval = []; yval = []

for t in range(50):
    s = complex(0.5,t)
    xval.append(s)
    sum = 0
    for n in range(1, 1000):
        sum += 1/n**s

    yval.append(sum)

plt.plot(xval, yval) 
plt.savefig('riemann.pdf')
