#!/usr/bin/env python3
# File: fib.py
# Name: D.Saravanan
# Date: 14/07/2021
# Fibonacci number calculation

import ctypes

def fib(num):

    if num <= 0: return -1
    elif num == 1: return 0
    elif (num == 2 or num == 3): return 1
    else: return fib(num-2) + fib(num-1)


_libfib = ctypes.CDLL('./_fib.so')

def ctypes_fib(num):
    return _libfib.fib(ctypes.c_int(num))
