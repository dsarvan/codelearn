#!/usr/bin/env python3
# File: evenodd.py
# Name: D.Saravanan
# Date: 22/09/2020
# Script to find a number is even/odd and print the next ten even/odd numbers

def nextten(num):
    """ print next ten odd/even numbers """
    for value in range(num, num+20, 2):
        print(value)

def condition(numb):
    """ check conditions """
    if numb%2 == 0:
        print('\n{} is even number'.format(int(numb)))
        nextten(int(numb))
    elif numb%2 != 0:
        print('\n{} is odd number'.format(int(numb)))
        nextten(int(numb))
    else:
        print('Error message: Invalid input.')

if __name__ == '__main__':

    number = float(input('Enter number: '))

    if number.is_integer():
        condition(number)
    else:
        print('Error message: Invalid input.')
