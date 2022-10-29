#!/usr/bin/env python3
# File: genexpr.py
# Name: D.Saravanan
# Date: 28/07/2021

""" Script to prefer a generator expression to a list comprehension for simple iteration
    Reference: Writing Idiomatic Python """

institute = ["institute of mathematical sciences", "raman research institute",\
             "chennai mathematical institute", "physical research laboratory",\
             "indian institute of science education and research"]

# harmful list comprehension
for uname in [name.upper() for name in institute]:
    print(len(uname))

# idiomatic generator expression
for uname in (name.upper() for name in institute):
    print(len(uname))
