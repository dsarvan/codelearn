#!/usr/bin/env python
# File: perfectsquare.py
# Name: D.Saravanan
# Date: 20/10/2022

""" Test whether a number is a perfect square """

def is_square(num):
	root = int(num**0.5)
	return num == root*root

print(is_square(49))
print(is_square(50))
