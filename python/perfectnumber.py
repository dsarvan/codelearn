#!/usr/bin/env python
# File: perfectnumber.py
# Name: D.Saravanan
# Date: 18/03/2023

""" Script to find a perfect number """


def main():
    n = 10000
    for x in range(2, n + 1):
        sum = 0
        for n in range(1, x // 2 + 1):
            if x % n == 0:
                sum = sum + n

        if x == sum:
            print(x)


if __name__ == "__main__":
    main()
