#!/usr/bin/env python
# File: argparsedefault.py
# Name: D.Saravanan
# Date: 20/09/2024

""" Script that defines default values in optional arguments """

# Optional arguments can have default values that are used when the argument
# is not provided in the command line. You can set a default value using the
# default parameter.

import argparse

# initialize the ArgumentParser
parser = argparse.ArgumentParser(description="A program to demonstrate default values for optional arguments.")

# define optional argument with default value
parser.add_argument("-n", "--name", default="Berg", help="Specify your name.")

# parse the arguments
args = parser.parse_args()

# accessing the parsed argument
print(f"Hello, {args.name}.")

# If you run python argparsedefault.py without any arguments, it will print
# Hello, Berg. because Berg is the default value for the --name option.
