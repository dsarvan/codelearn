#!/usr/bin/env python
# File: fractals.py
# Name: D.Saravanan
# Date: 29/10/2022

""" Script for fractals animation """

import turtle

def create_l_system(iters, axiom, rules):
	start_string = axiom
	if iters == 0:
		return axiom
	end_string = ""
	for _ in range(iters):
		end_string = "".join(rules[i] if i in rules else i for i in start_string)
		start_string = end_string

	return end_string

def draw_l_system(t, instructions, angle, distance):
	for cmd in instructions:
		if cmd == 'F':
			t.forward(distance)
		elif cmd == '+':
			t.right(angle)
		elif cmd == '-':
			t.left(angle)

def main(iterations, axiom, rules, angle, length=8, size=2, y_offset=0, x_offset=0, offset_angle=0, width=450, height=450):
	
	inst = create_l_system(iterations, axiom, rules)

	t = turtle.Turtle()
	wn = turtle.Screen()
	wn.setup(width, height)

	t.up()
	t.backward(-x_offset)
	t.left(90)
	t.backward(-y_offset)
	t.left(offset_angle)
	t.down()
	t.speed(0)
	t.pensize(size)
	t.hideturtle()
	draw_l_system(t, inst, angle, length)

	wn.exitonclick()

if __name__ == "__main__":

	# Koch-Snowflake
	#axiom = "F--F--F"
	#rules = {"F":"F+F--F+F"}
	#iterations = 4
	#angle = 60

	# Quadratic-Koch-Island
	#axiom = "F+F+F+F"
	#rules = {"F":"F-F+F+FFF-F-F+F"}
	#iterations = 2
	#angle = 90

	# Crystal
	axiom = "F+F+F+F"
	rules = {"F":"FF+F++F+F"}
	iterations = 3
	angle = 90

	#axiom = "FXF--FF--FF"
	#rules = {"F":"FF", "X":"--FXF++FXF++FXF--"}
	#iterations = 7
	#angle = 60

	#axiom = "F+F+F+F"
	#rules = {"F":"FF+F+F+F+FF"}
	#iterations = 3
	#angle = 90

	# Tiles
	#axiom = "F+F+F+F"
	#rules = {"F":"FF+F-F+F+FF"}
	#iterations = 3
	#angle = 90

	main(iterations, axiom, rules, angle)
