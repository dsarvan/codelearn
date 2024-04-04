#!/usr/bin/env python
# File: gaussian.py
# Name: D.Saravanan
# Date: 17/03/2024

""" Script to generate isolated Gaussian pulse """

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("classic")
plt.rc("text", usetex="True")
plt.rc("pgf", texsystem="pdflatex")
plt.rc("font", family="serif", weight="normal", size=10)
plt.rc("axes", labelsize=12, titlesize=12)
plt.rc("figure", titlesize=12)


def gaussian(fs: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate isolated Gaussian pulse.

    :param fs: sampling frequency in Hz
    :param sigma: pulse width in seconds

    :returns: (t,g): time base (t) and the signal g(t)
    :rtype: tuple[np.ndarray, np.ndarray]

    :example:

    >>> fs = 80
    >>> sigma = 0.1
    >>> t, g = gaussian(fs, sigma)

    """

    t = np.linspace(-0.5, 0.5, fs)
    g = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * (t / sigma) ** 2)
    return t, g


def main():

    fs = 80
    sigma = 0.1
    t, g = gaussian(fs, sigma)

    fig, ax = plt.subplots()
    ax.plot(t, g, "r", lw=1, label="gaussian")
    ax.grid(True, which="both")
    ax.set(xlim=(t[0], t[-1]), ylim=(0, 4.5))
    ax.set(xlabel="Time(s)", ylabel="Amplitude")
    ax.set_title(r"Gaussian pulse $\sigma = 0.1 s$")
    plt.savefig("gaussian.png", dpi=100)


if __name__ == "__main__":
    main()
