#!/usr/bin/env python
# File: distmetrics.py
# Name: D.Saravanan
# Date: 31/03/2023

""" Script to compute distance metrics """

from scipy.spatial import distance


def euclidean(x, y):
    """Euclidean distance is the shortest
    distance between any two points in a
    metric space."""
    return distance.euclidean(x, y)


def manhattan(x, y):
    """Manhattan distance, also called
    taxicab distance or cityblock distance."""
    return distance.cityblock(x, y)


def minkowski(x, y):
    """Minkowski distance equation takes the same
    form as that of Manhattan distance for p = 1.
    Similarly, for p = 2, the Minkowski distance
    is equivalent to the Euclidean distance."""
    return distance.minkowski(x, y, p=2)


if __name__ == "__main__":
    x = [3, 6, 9]
    y = [1, 0, 1]

    d1 = euclidean(x, y)
    print(f"The Euclidean distance between the points {x} and {y}: {d1}")

    d2 = manhattan(x, y)
    print(f"The Manhattan distance between the points {x} and {y}: {d2}")

    d3 = minkowski(x, y)
    print(f"The Minkowski distance between the points {x} and {y}: {d3}")
