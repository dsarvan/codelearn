#!/usr/bin/env python3
# File: isfactor.py
# Name: D.Saravanan
# Date: 23/09/2020
# Script to calculate factor of an integer

def is_factor(a, b):
    if a%b == 0:
        print('True, {} is a factor of {}'.format(b, a))
    else:
        print('False, {} is not a factor of {}'.format(b, a))

a = int(input('Enter numerator: '))
b = int(input('Enter denominator: '))

is_factor(a, b)
