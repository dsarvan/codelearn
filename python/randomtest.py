#!/usr/bin/env python3
# File: randomtest.py
# Name: D.Saravanan
# Date: 25/10/2020
# Script to generate random number of size 6 and count the number again the same number generates

import numpy as np

count = 0
r1 = np.random.randint(1, 7, 6)

while True:
    r2 = np.random.randint(1, 7, 6)
    
    if np.array_equal(r1, r2) == False:
        count += 1
    else:
        break

print("r1: {}, r2: {}, count: {}".format(r1, r2, count))
