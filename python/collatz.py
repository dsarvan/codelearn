#!/usr/bin/env python
# File: collatz.py
# Name: D.Saravanan
# Date: 02/05/2024

""" Script for 3n + 1 collatz conjecture """

import matplotlib.pyplot as plt

plt.style.use("classic")
plt.rc("text", usetex="True")
plt.rc("figure", titlesize=12)
plt.rc("pgf", texsystem="pdflatex")
plt.rc("axes", labelsize=12, titlesize=12)
plt.rc("font", family="serif", weight="normal", size=10)


def conjecture(n: int) -> tuple[int, list[int]]:
    """collatz conjecture"""
    sequence: list[int] = [n]
    iteration: int = 0

    while n != 1:
        n = n // 2 if n % 2 == 0 else (3 * n) + 1
        sequence = sequence + [n]
        iteration = iteration + 1

    return iteration, sequence


if __name__ == "__main__":

    num_iter: int
    seq_value: list[int]
    initial_value: int = 97

    num_iter, seq_value = conjecture(initial_value)

    fig, ax = plt.subplots()
    ax.plot(seq_value, "r", lw=1)
    ax.grid(True, which="both")
    ax.tick_params(which="major", direction="inout")
    ax.set(xlabel="Iteration", ylabel="Iterate Value")
    ax.set(xlim=(0, num_iter), ylim=(min(seq_value), max(seq_value)))
    ax.text(
        0.7,
        0.9,
        f"Initial Value: {initial_value}\n Number of Iterations: {num_iter}",
        ha="left",
        va="bottom",
        transform=ax.transAxes,
    )
    ax.set_title(r"Collatz Conjecture")
    plt.savefig("collatz.png", dpi=100)
