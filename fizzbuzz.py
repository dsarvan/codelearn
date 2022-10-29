#!/usr/bin/env python3

for n in range(1, 101):
    if (n%3 == 0) or (n%5 == 0):
        print("fizz")
    elif (n%3 == 0) and (n%5 == 0):
        print("buzz")
    else:
        print(n)
