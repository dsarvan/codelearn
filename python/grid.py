#!/usr/bin/env python3
# File: grid.py
# Name: D.Saravanan
# Date: 30/03/2021
# Script to draw grid using turtle module

import turtle

# configuration
screen = turtle.Screen()
turtle.setup(1000,1000)
turtle.title("Conway's Game of Life")
turtle.hideturtle()
turtle.speed(0)
turtle.tracer(0,0)

n = 50 # 50 x 50 grid
def draw_line(x1,y1,x2,y2):
    """ this function draw a line between (x1,y1) and (x2,y2) """
    turtle.up()
    turtle.goto(x1,y1)
    turtle.down()
    turtle.goto(x2,y2)

def draw_grid():
    """ this function draws n X n grid """
    turtle.pencolor('gray')
    turtle.pensize(3)

    x = -400
    for i in range(n+1):
        draw_line(x,-400,x,-400)
        x += 800/n

    y = -400
    for i in range(n+1):
        draw_line(-400,y,-400,y)
        y += 800/n

draw_grid()
screen.update()
turtle.done()
