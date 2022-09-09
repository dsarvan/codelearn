#!/usr/bin/env python

if __name__ == "__main__":
    # not possible as lambda
    def if_buzz(x):
        if x % 15 == 0:
            print("fizzBuzz")
        elif x % 3 == 0:
            print("Fizz")
        elif x % 5 == 0:
            print("Buzz")
        else:
            print(x)

    def fizzbuzz(x):
        match x % 3, x % 5:
            case 0, 0: print("FizzBuzz"),
            case 0, _: print("Fizz"),
            case _, 0: print("Buzz"),
            case _, _: print("{}", x)

    list(map(if_buzz, range(1, 24)))
    list(map(fizzbuzz, range(1, 24)))
