#!/usr/bin/env python
# File: functional3.py
# Name: D.Saravanan
# Date: 09/05/2023

""" Script for immutable function argument """


# object-oriented
def addValue(items, value):
    """argument is mutated"""
    items.append(value)


items = [1, 2, 3]
print(items)
addValue(items, 5)
print(items)


# functional
def addValue(items, value):
    """argument not mutated"""
    return items + [value]


items = [1, 2, 3]
print(items)
addValue(items, 5)
print(items)
