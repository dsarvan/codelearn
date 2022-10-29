#!/usr/bin/env python3
# File: largest.py
# Name: D.Saravanan
# Date: 03/05/2020
# Script to return the largest number from given two integer arrays of an integer length

def maxNum(n, m, k):
    ''' maximum number from two lists '''
    n.extend(m); value = []
    for _ in range(k):
        nvalue = max(n)
        value.append(nvalue)
        n.remove(nvalue)

    result = sum(num*10**i for i, num in enumerate(value[::-1]))
    return result

if __name__ == '__main__':

    n = [3,4,6,5]
    m = [9,0,2,5,8,3]
    k = 5

    print(maxNum(n, m, k))
