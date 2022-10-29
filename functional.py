#!/usr/bin/env python3
# File: functional.py
# Name: D.Saravanan
# Date: 30/09/2020

""" Script to explain imperative, object-oriented, functional programming """


# Imperative Programming
s = 0
for n in range(1, 10):
    if n % 3 == 0 or n % 5 == 0:
        s += n
print(s)

del s

# Object-Oriented Programming
s = []
for n in range(1, 10):
    if n % 3 == 0 or n % 5 == 0:
        s.append(n)
print(sum(s))

del s

# Functional Programming
s = [n for n in range(1, 10) if n % 3 == 0 or n % 5 == 0]
print(sum(s))

del s

# using generator
s = (n for n in range(1, 10) if n % 3 == 0 or n % 5 == 0)
print(sum(s))
