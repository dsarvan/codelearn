#!/usr/bin/env python3
# File: series.py
# Name: D.Saravanan
# Date: 08/08/2021

""" Script to print Fibonacci series """

def fibonacci(fib1, fib2, num):

    """ fibonacci series generation """

    while num <= 51:
        print(fib1)
        fib1, fib2 = fib2, fib1+fib2
        num += 1

fibonacci(0, 1, 1)
