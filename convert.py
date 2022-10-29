#!/usr/bin/env python3
# File: convert.py
# Name: D.Saravanan
# Date: 23/09/2020
# Script to unit conversion

""" 1 inch = 2.54 cm """
inch = float(input('\nEnter length in inches: '))
print('{} inch = {} cm'.format(inch, inch*2.54))

""" 1 mile = 1.609 kilometers """
mile = float(input('\nEnter distance in miles: '))
print('{} mile = {} km'.format(mile, mile*1.609))

""" Fahrenheit to Celsius """
F = float(input('\nTemperature in Fahrenheit: '))
C = (F - 32) * (5/9)
print('{:.1f} degrees Fahrenheit = {:.1f} degrees Celsius'.format(F, C))

""" Celsius to Fahrenheit """
C = float(input('\nTemperature in Celsius: '))
F = (C * (9/5)) + 32
print('{:.1f} degrees Celsius = {:.1f} degrees Fahrenheit'.format(C, F))
