#!/usr/bin/env python3
# File: conway.py
# Name: D.Saravanan
# Date: 11/03/2021
# Script to implement Conway's Game of Life

"""
Let us define the eight grid squares around a cell as its neighbours. The rules of Life
are then as follows:
    1. A living cell will survive into the next generation by default, unless:
        * it has fewer than two live neighbours (underpopulation)
        * it has more than three live neighbours (overpopulation)
    2. A dead cell will spring to life if it has exactly three live neighbours 
"""

import numpy as np
import matplotlib
matplotlib.use('TKAgg', force=True)
import matplotlib.pyplot as plt

universe = np.zeros((6, 6))

beacon = [[1, 1, 0, 0],
          [1, 1, 0, 0],
          [0, 0, 1, 1],
          [0, 0, 1, 1]]

universe[1:5, 1:5] = beacon

if universe[x, y] and not 2 <= num_neighbours <= 3:
    new_universe[x, y] = 0
elif num_neighbours == 3:
    new_universe[x, y] = 1

universe = new_universe

plt.imshow(universe, cmap='binary')
plt.show()
