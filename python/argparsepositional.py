#!/usr/bin/env python
# File: argparsepositional.py
# Name: D.Saravanan
# Date: 18/09/2024

""" Script that defines a positional arguments """

# In command-line programs, positional arguments are those that must be entered in a
# specific order, without a preceding option flag. They are called "positional" because
# their meaning is determined by their position in the command line. For example, in a
# command like cp source_file destination_file, source_file and destination_file are
# positional arguments. In the context of Python's argparse library, positional
# arguments are required by default, and their order matters.

import argparse

# initialize the ArgumentParser
parser = argparse.ArgumentParser(description="A program to read a file.")

# define a positional argument "filename"
parser.add_argument("filename", help="The name of the file to read.")

# parse the arguments
args = parser.parse_args()

# accessing the parsed argument
print(f"Reading file {args.filename}.")

# In this example, running python argparsepositional.py filename.txt will produce the
# output Reading file filename.txt. Notice that filename.txt is a positional argument
# because it doesn't have a preceding flag and its position matters.
