#!/usr/bin/env python3
# File: binary.py
# Name: D.Saravanan
# Date: 31/05/2021
# Script to convert number from base 10 to base 2

import numpy as np

num = int(input("Enter number (base 10): "))

rval = []
qval = int(num/2)

while qval != 0:
    rval.append(int(num%2))
    qval = int(num/2)
    num = qval

rval.reverse()
print(np.array(rval))
