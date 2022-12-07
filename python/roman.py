#!/usr/bin/env python
# File: roman.py
# Name: D.Saravanan
# Date: 08/12/2022

""" Script to convert roman numerals into decimal numbers """

import re

d = {'M':1000, 'CM':900, 'D':500, 'CD':400, 'C':100, 'XC':90, 'L':50, 'XL':40, 'X':10, 'IX':9, 'V':5, 'IV':4, 'I':1}

pattern = re.compile(r"""(?x)
						 (M{0,3})(CM)?
						 (CD)?(D)?(C{0,3})
						 (XC)?(XL)?(L)?(X{0,3})
						 (IX)?(IV)?(V)?(I{0,3})""")

num = input('Enter roman numeral: ').upper()
m = pattern.match(num)

sum = 0
for x in m.groups():
	if x != None and x != '':
		if x in ['CM', 'CD', 'XC', 'XL', 'IX', 'IV']:
			sum += d[x]
		elif x[0] in 'MDCLXVI':
			sum += d[x[0]] * len(x)

print(sum)
