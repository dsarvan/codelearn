#!/usr/bin/env python
# File: rectobject.py
# Name: D.Saravanan
# Date: 06/05/2023

"""Script to create class of objects as Rectangle"""


class Rectangle:
    """A Python object that describes the properties of a rectangle."""

    def __init__(self, width, height, center=(0.0, 0.0)):
        """Sets the attributes of a particular instance of 'Rectangle'.

        width: float
            The x-extent of this rectangle instance.

        height: float
            The y-extent of this rectangle instance.

        center: Tuple[float, float], optional (default = (0, 0))
            The (x, y) position of this rectangle's center."""

        self.width = width
        self.height = height
        self.center = center

    def __repr__(self):
        """Returns a string to be used as a printable representation
        of a given rectangle."""
        w = self.width; h = self.height; c = self.center
        return f"Rectangle(width = {w}, height = {h}, center = {c})"

    def area(self):
        """Returns the area of this rectangle."""
        return self.width * self.height

    def corners(self):
        """Returns the (x, y) corner-locations of this rectangle, starting
        with the 'top-right' corner, and proceeding clockwise."""
        cx, cy = self.center; dx = self.width / 2.0; dy = self.height / 2.0
        return [(cx+x,cy+y) for x,y in ((dx,dy), (dx,-dy), (-dx,-dy), (-dx,dy))]


if __name__ == "__main__":
    rect = Rectangle(5, 10)
    print(rect)
    print(f"Area: {rect.area()}")
    print(f"Corners: {rect.corners()}")
