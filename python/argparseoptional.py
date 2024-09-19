#!/usr/bin/env python
# File: argparseoptional.py
# Name: D.Saravanan
# Date: 19/09/2024

""" Script that defines an optional (named) arguments """

# In contrast to positional arguments, optional arguments (also known as named
# arguments) are not required and can appear in any order in the command line.
# These arguments are usually specified by short or long option names that
# precede the argument value. Optional arguments allow for greater flexibility
# and can have default values.
#
# Short options: Single-letter options prefixed with a single dash (example: -f).
# Long options: Multi-letter options prefixed with two dashes (example: --foo).
#
# You can define either or both for each optional argument.

import argparse

# initialize the ArgumentParser
parser = argparse.ArgumentParser(description="A program to demonstrate optional arguments.")

# define optional arguments
parser.add_argument("-s", "--short", help="This is a short option.")
parser.add_argument("-l", "--long", help="This is a long option.")

# parse the arguments
args = parser.parse_args()

# accessing the parsed arguments
if args.short:
	print(f"Short option: {args.short}")

if args.long:
	print(f"Long option: {args.long}")

# In this example, running
# python argparseoptional.py -s value1 --long=value2
# will produce:
# Short option: value1
# Long Option: value2
