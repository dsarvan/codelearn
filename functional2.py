#!/usr/bin/env python3
# File: functional2.py
# Name: D.Saravanan
# Date: 30/09/2021

""" Script using map, filter and lambda function """


def gt10(x):
    return x > 19


print(map(gt10, [1, 3, 18, 20]))

print(filter(gt10, [1, 3, 18, 20]))

print(sum(filter(gt10, [1, 3, 18, 20])))

print(sum(filter(lambda x: x > 10, [1, 3, 18, 20])))
