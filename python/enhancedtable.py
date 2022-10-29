#!/usr/bin/env python3
# File: enhancedtable.py
# Name: D.Saravanan
# Date: 25/09/2020
# Script for enhanced multiplication table generator

def table(tnum, rstt, rstp):
    """ multiplication table generator """
    for idx in range(rstt, rstp+1):
        print('{} x {} = {}'.format(tnum, idx, tnum*idx))

if __name__ == '__main__':

    tnumber = float(input('Enter number for which table generated: '))
    rstart = float(input('Enter start range value: '))
    rstop = float(input('Enter stop range value: '))

    if tnumber > 0 and tnumber.is_integer():
        if rstart > 0 and rstart.is_integer():
            if rstop > 0 and rstop.is_integer():
                table(int(tnumber), int(rstart), int(rstop))
    else:
        print('Error message: Invalid input')
