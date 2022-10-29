#!/usr/bin/env python3
# File: complexn.py
# Name: D.Saravanan
# Date: 23/09/2020
# Script using complex function

try:
    z = complex(input('Enter complex number: '))
    print('Entered:', z)
    print('real part of {}: {}'.format(z, z.real))
    print('imagninary part of {}: {}'.format(z, z.imag))
except ValueError:
    print('Invalid input, No spaces between real and imaginary part.')
