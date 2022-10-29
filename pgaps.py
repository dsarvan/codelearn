#!/usr/bin/env python3
# File: pgaps.py
# Name: D.Saravanan
# Date: 12/10/2020
# Script to find Small and Large gaps between primes

import sympy as sy
import matplotlib.pyplot as plt

N = 10000000
primes = list(sy.primerange(1, N))

gaps = []
for n in range(1, len(primes)):
    gaps.append(primes[n] - primes[n-1])

    plt.plot(primes[n], primes[n-1], 'ob')

print('Large gap: {}'.format(max(gaps)))
print('Small gap: {}'.format(min(gaps)))

plt.show()
