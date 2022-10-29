#!/usr/bin/env python3
# File: subplots.py
# Name: D.Saravanan
# Date: 06/07/2021

""" Script to plot five subplots with zero vertical spacing """

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.use('TkAgg')

NROWS = 5
fig, axes = plt.subplots(NROWS, 1)

# zero vertical space between subplots
fig.subplots_adjust(hspace=0)

x = np.linspace(0, 1, 1000)

for i in range(NROWS):
    # n = NROWS for the top subplot, n = 0 for the bottom subplot
    n = NROWS - i
    
    axes[i].plot(x, np.sin(n * np.pi * x), 'k', lw=2)

    # we only want ticks on the bottom of each subplot
    axes[i].xaxis.set_ticks_position('bottom')

    if i < NROWS-1:
        # set ticks at the nodes (zeros) of our sine functions
        axes[i].set_xticks(np.arange(0, 1, 1/n))

        # we only want labels on the bottom subplot x-axis
        axes[i].set_xticklabels('')

    axes[i].set_yticklabels('')

plt.savefig('subplots.png', dpi=300)
