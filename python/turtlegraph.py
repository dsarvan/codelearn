#!/usr/bin/env python
# File: turtlegraph.py
# Name: D.Saravanan
# Date: 05/05/2023

""" Script for turtle graphics """

import turtle


def drawSquare(xpos, ypos, length):
    """4 sides = range(4) and ttl.right(90)
     8 sides = range(8) and ttl.right(45)
    12 sides = range(12) and ttl.right(30)"""

    ttl.penup()
    ttl.goto(xpos, ypos)
    ttl.setheading(0)
    ttl.pendown()
    for _ in range(4):
        ttl.forward(length)
        ttl.right(90)
    ttl.penup()


if __name__ == "__main__":
    ttl = turtle.Turtle()
    ttl.hideturtle()
    ttl.speed(1)
    ttl.pensize(3)
    drawSquare(-50, 50, 100)

    turtle.done()
