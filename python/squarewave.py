#!/usr/bin/env python
# File: squarewave.py
# Name: D.Saravanan
# Date: 26/08/2022

""" Script to create square wave by summing sine wave """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

plt.style.use("classic")
plt.rc("text", usetex="True")
plt.rc("figure", titlesize=10)
plt.rc("pgf", texsystem="pdflatex")
plt.rc("axes", labelsize=10, titlesize=10)
plt.rc("font", family="serif", weight="normal", size=8)


def formatfunc(value, tick_number):
    """set major axis formatter"""
    nval = int(np.round(2 * value / np.pi))
    if nval == 0:
        return r"$0$"
    if nval % 2 > 0 or nval % 2 < 0:
        return r"${0}\pi/2$".format(nval)
    return r"${0}\pi$".format(nval // 2)


if __name__ == "__main__":
    xval = np.linspace(-2 * np.pi, 2 * np.pi, 5000, endpoint=True)
    square_wave = np.sin(xval)

    fwriter = animation.writers["ffmpeg"]
    data = {"title": "Square wave animation"}
    writer = fwriter(fps=15, metadata=data)

    fig, ax = plt.subplots()
    (line1,) = ax.plot(xval, square_wave, c="#363737", lw=1)
    ax.spines[["left", "bottom"]].set_position(("data", 0))
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_yticks([-1.5, -1, -0.5, 0.5, 1, 1.5])
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(formatfunc))
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(which="major", direction="inout")
    ax.set(xlim=(-2 * np.pi, 2 * np.pi), ylim=(-1.5, 1.5))
    ax.set_title(r"Square wave by summing sine wave", pad=20)

    with writer.saving(fig, "squarewave.mp4", dpi=300):
        line1.set_ydata(square_wave)
        writer.grab_frame()

        for n in range(3, 10001, 2):
            square_wave += (1 / n) * np.sin(n * xval)

            line1.set_ydata(square_wave)
            writer.grab_frame()
