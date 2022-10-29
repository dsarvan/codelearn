#!/usr/bin/env python3
# File: guess.py
# Name: D.Saravanan
# Date: 16/08/2020
# Script for number guessing game

import numpy as np

option = 'yes'

while option == 'yes':

    N = np.random.randint(1, 100, size=1)

    num = int(input("\nI 'm thinking of a number! Try to guess the number I 'm thinking of: "))

    while num != N:

        if num < N:
            num = int(input("Too low! Guess again: "))
        elif num > N:
            num = int(input("Too high! Guess again: "))

    if num == N:
        option = input("That's it! Would you like to play again? (yes/no) ")

print("Thanks for playing!")
