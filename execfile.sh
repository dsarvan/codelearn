#!/usr/bin/env bash
# File: execfile.sh
# Name: D.Saravanan
# Date: 15/09/2020
# Script to find the executable files

echo "List of executable files in the $1 directory:"

# find all executable files in the current directory
# ./execfile
#find . -perm -u+x -type f

# find all executable file in the directory given as argument
# ./execfile.sh ~/Documents
find $1 -perm -u+x -type f
