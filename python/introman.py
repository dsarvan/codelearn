#!/usr/bin/env python3
# File: introman.py
# Name: D.Saravanan
# Date: 20/01/2021
# Script to convert integer to roman numerals

conversion = {1: 'I', 4: 'IV', 5: 'V', 9: 'IX', 10: 'X', 40: 'XL', 50: 'L', 90:
        'XC', 100: 'C', 400: 'XD', 500: 'D', 900: 'CM', 1000: 'M'}

integer = int(input("Enter an integer: "))

numbers = [1000, 900, 500, 400, 100, 50, 40, 10, 9, 5, 4, 1]

for n in numbers:
    if integer > 0: 
        quotient = integer//n

        if quotient != 0:
            for val in range(quotient):
                print(conversion[n], end='')

        integer = integer%n
