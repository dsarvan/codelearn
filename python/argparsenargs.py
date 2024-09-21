#!/usr/bin/env python
# File: argparsenargs.py
# Name: D.Saravanan
# Date: 21/09/2024

""" Script that defines a required and an optional positional arguments """

# In argparse, positional arguments are required by default, meaning that the program
# will produce an error if you don't provide them. However, you can make a positional
# argument optional by setting the nargs parameter to "?".

import argparse

# initialize the ArgumentParser
parser = argparse.ArgumentParser(description="A program to demonstrate optional positional arguments.")

# define a required positional argument "required_arg"
parser.add_argument("required_arg", help="This argument is required.")

# define an optional positional argument "optional_arg"
parser.add_argument("optional_arg", nargs="?", default="default_value", help="This argument is optional.")

# parse the arguments
args = parser.parse_args()

# accessing the parsed arguments
print(f"Required argument: {args.required_arg}")
print(f"Optional argument: {args.optional_arg}")

# In this case, running the program without the optional positional argument will not
# result in an error, and the value for optional_arg will be set to its default value,
# which is default_value in this case.
