#!/usr/bin/env python3
# File: fibonacci.py
# Name: D.Saravanan
# Date: 31/05/2021

""" Script to compute nth value of the Fibonacci series """

def fibonacci(nval):
    """ fibonacci function """
    return 1 if nval in (1, 2) else fibonacci(nval-1) + fibonacci(nval-2)

if __name__ == "__main__":
    N = 10
    value = fibonacci(N)
    print("The value of the Fibonacci series (for n = %d) is %d." %(N, value))
