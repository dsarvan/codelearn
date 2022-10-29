#!/usr/bin/env python3
# File: classdef2.py
# Name: D.Saravanan
# Date: 31/05/2021

""" Script to explain user-defined classes """

class Worker:
    """ worker class """

    def __init__(self, name, pay):          # initialize when created
        """ self function """
        self.name = name                    # self is the new object
        self.pay = pay

    def last_name(self):
        """ worker last name """
        return self.name.split()[-1]        # split string on blanks

    def give_raise(self, percent):
        """ worker pay raise """
        self.pay *= (1.0 + percent)         # update pay in place

ram = Worker('Raghu Raman', 50000)          # make two instances
ksh = Worker('Anantha Krishnan', 60000)     # each has name and pay attrs

print(ram.last_name())                       # call method: ram is self
print(ksh.last_name())                       # ksh is the self object
ksh.give_raise(.10)                          # update ksh's pay
print(ksh.pay)
