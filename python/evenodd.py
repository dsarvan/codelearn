#!/usr/bin/env python3

def nextten(number):
    """ print next ten odd/even numbers """

    for value in range(number, number+20, 2):
        print(value)


def condition(number):
    """ check conditions """

    if number%2 == 0:
        print('\n{} is even number'.format(int(number)))
        nextten(int(number))

    elif number%2 != 0:
        print('\n{} is odd number'.format(int(number)))
        nextten(int(number))

    else:
        print('Error message: Invalid input.')


if __name__ == '__main__':

    number = float(input('Enter number: '))

    if number.is_integer():
        condition(number)

    else:
        print('Error message: Invalid input.')
