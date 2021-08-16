#!/usr/bin/env python3
# File: numwords.py
# Name: D.Saravanan
# Date: 16/08/2021

""" Script to convert number to words """

import inflect
import num2word

p = inflect.engine()

number = int(input("Enter number: "))
print(p.number_to_words(number))

print(num2word.to_card(number))
