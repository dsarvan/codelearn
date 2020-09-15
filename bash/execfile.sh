#!/usr/bin/env bash
# File: execfile.sh
# Name: D.Saravanan
# Date: 15/09/2020
# Script to find the executable files

echo "List of executable files in the $1 directory:"
#find . -perm -u+x -type f
find $1 -perm -u+x -type f
