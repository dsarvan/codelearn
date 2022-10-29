#!/usr/bin/env python3
# File: enhancedconvert.py
# Name: D.Saravanan
# Date: 25/09/2020
# Script to enhanced unit converter

def section():
    """ base values """
    print('\nSelect the option: \n\
    1. Lenght converter \n\
    2. Mass converter \n\
    3. Temperature converter')

    option1 = input('\nEnter your option: ')

    if option1 == '1':
        length()
    elif option1 == '2':
        mass()
    else:
        temperature()

def length():
    """ 1 mile = 1.6093 kilometer """
    print('\nLenght converter: \n\
    1. kilometers to miles \n\
    2. miles to kilometers')

    option2 = input('\nEnter your option: ')

    if option2 == '1':
        kilometer = float(input('Enter distance in kilometers: '))
        print('{} kilometer = {} mile'.format(kilometer, kilometer/1.6093))
    else:
        mile = float(input('Enter distance in miles: '))
        print('{} mile = {} kilometer'.format(mile, mile*1.6093))

def mass():
    """ 1 kilogram = 2.2046 pound """
    print('\nMass converter: \n\
    1. kilograms to pounds \n\
    2. pounds to kilograms')

    option2 = input('\nEnter your option: ')

    if option2 == '1':
        kilogram = float(input('Enter mass in kilograms: '))
        print('{} kilogram = {} pound'.format(kilogram, kilogram*2.2046))
    else:
        pound = float(input('Enter mass in pounds: '))
        print('{} pound = {} kilogram'.format(pound, pound/2.2046))

def temperature():
    """ 1 Celsius = 33.8 Fahrenheit """
    print('\nTemperature converter: \n\
    1. Celsius to Fahrenheit \n\
    2. Fahrenheit to Celsius')

    option2 = input('\nEnter your option: ')

    if option2 == '1':
        celsius = float(input('Enter temperature in Celsius: '))
        print('{} degree Celsius = {} degree Fahrenheit'.format(celsius, celsius*(33.8)))
    else:
        fahrenheit = float(input('Enter temperature in Fahrenheit: '))
        print('{} degree Fahrenheit = {} degree Celsius'.format(fahrenheit, fahrenheit/(33.8)))


if __name__ == '__main__':
    section()
