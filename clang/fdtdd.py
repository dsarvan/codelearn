#!/usr/bin/env python
from sys import argv
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("classic")
plt.rc("text", usetex="True")
plt.rc("pgf", texsystem="pdflatex")
plt.rc("font", family="serif", weight="normal", size=10)
plt.rc("axes", labelsize=12, titlesize=12)
plt.rc("figure", titlesize=12)

data = np.genfromtxt(argv[1])
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle(r"$FDTD\ simulation\ of\ a\ pulse\ in\ free\ space$")
ax1.plot(data[:,0], data[:,1], 'k', lw=1)
ax1.set(xlim=(0,200), ylim=(-1.2,1.2), ylabel=r"$E_x$")
ax1.set(xticks=np.arange(0,220,20), yticks=np.arange(-1,1.2,1))
ax2.plot(data[:,0], data[:,2], 'k', lw=1)
ax2.set(xlim=(0,200), ylim=(-1.2,1.2), xlabel=r"$FDTD\ cells$", ylabel=r"$H_y$")
ax2.set(xticks=np.arange(0,220,20), yticks=np.arange(-1,1.2,1))
plt.subplots_adjust(bottom=0.2, hspace=0.45)
plt.savefig("fdtdd.png", dpi=200)
