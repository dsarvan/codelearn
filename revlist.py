#!/usr/bin/env python3
# File: revlist.py
# Name: D.Saravanan
# Date: 03/08/2021

""" Script to reverse a list in python """

my_list = [1, 2, 3, 4, 5, 6]

# reverse the list in-place
my_list.reverse()

print(my_list)

my_list = [1, 2, 3, 4, 5, 6]

# reverse a list using slice notation
segment_1 = my_list[1:5:1]
segment_2 = my_list[1:5:2]

print(segment_1)
print(segment_2)

reversed_list = my_list[::-1]

print(reversed_list)
