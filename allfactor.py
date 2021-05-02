#!/usr/bin/env python3
# File: allfactor.py
# Name: D.Saravanan
# Date: 23/09/2020
# Script to find all factors of an integer

value = []

def factors(N):
    ''' factors of an integer '''
    for n in range(1, N+1):
        if N%n == 0:
            value.append(n)

    print('The factors of {} are {}'.format(N, value))

if __name__=='__main__':

    N = float(input('Enter number: '))

    if N > 0 and N.is_integer():
        factors(int(N))
    else: 
        print('Invalid input, enter a positive integer.')
