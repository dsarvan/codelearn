#!/usr/bin/env python3

def table(N):
    for n in range(1,11):
        print('{} x {} = {}'.format(N, n, N*n))


if __name__ == '__main__':

    N = float(input('Enter number: '))

    if N > 0 and N.is_integer():
        table(int(N))
    else:
        print('Invalid input')
