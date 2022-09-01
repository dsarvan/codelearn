#!/usr/bin/env python

if __name__ == "__main__":
    # not possible as lambda
    def fizzbuzz(x):
        if x % 15 == 0:
            print("FizzBuzz")
        elif x % 3 == 0:
            print("Fizz")
        elif x % 5 == 0:
            print("Buzz")
        else:
            print(x)

    for n in range(1, 16):
        fizzbuzz(n)

    print("---")
    list(map(fizzbuzz, range(1, 16)))
