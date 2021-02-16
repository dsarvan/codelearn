#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-6, 2, 10000)
y = [x**4 + 7*x**3 + 5*x**2 - 17*x + 3 for x in x]

plt.plot(x, y)
plt.xlim(-6, 2); plt.ylim(-60, 60)
plt.grid(True); plt.show()
