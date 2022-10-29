#!/usr/bin/env python3
# File: multitableloop.py
# Name: D.Saravanan
# Date: 27/09/2020
# Script to print multiplication table

def table(num):
    """ multiplication table """
    print('\nMulitplication table of {}:'.format(num))
    for value in range(1, 11):
        print('{} x {} = {}'.format(num, value, num*value))

if __name__ == '__main__':

    while True:
        number = float(input('\nEnter number: '))
        table(int(number))

        option = input('\nExit program (yes/no): ')

        if option == 'yes':
            break
        if option == 'no':
            continue
