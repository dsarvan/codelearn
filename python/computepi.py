#!/usr/bin/env python
# File: computepi.py
# Name: D.Saravanan
# Date: 06/03/2023

""" Script to compute pi """


def computepi(N):
    """compute pi"""
    dx = 1 / N  # step size
    x = lambda i: (i + 0.5) * dx
    pi = sum(map(lambda i: 4.0 / (1 + x(i) ** 2) * dx, range(N + 1)))
    return pi


if __name__ == "__main__":
    ns = 100000000  # number of steps
    pi = computepi(ns)
    print(pi)
