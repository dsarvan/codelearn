#!/usr/bin/env python3
# File: prime.py
# Name: D.Saravanan
# Date: 09/08/2021

""" Script that checks whether a number is prime """

import math

def prime(number):
    """ function to check prime """
    sqrt_number = math.sqrt(number)
    for n in range(2, int(sqrt_number) + 1):
        if (number/n).is_integer():
            return False
        return True

print(f'Check number(10,000,000) = {prime(10_000_000)}')
print(f'Check number(10,000,019) = {prime(10_000_019)}')

#def primevect(number):
#    """ function to check prime with concept of vectorization """
#    sqrt_number = math.sqrt(number)
#    numbers = range(2, int(sqrt_number) + 1)
#    for m in range(0, len(numbers), 5):
#        result = (number/numbers[m:(m+5)]).is_integer()
#        if any(result):
#            return False
#        return True
#
#print(f'Check number(10,000,000) = {primevect(10_000_000)}')
#print(f'Check number(10,000,019) = {primevect(10_000_019)}')
