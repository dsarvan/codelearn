#!/usr/bin/env python3
# File: classdef.py
# Name: D.Saravanan
# Date: 25/05/2021

""" Script for defining classes """

class Complex:
    """ complex number """

    def __init__(self, realpart, imagpart):
        """ assign arguments """
        self.real = realpart
        self.imag = imagpart

x = Complex(3.0, -4.5)
print(x.real, x.imag)
