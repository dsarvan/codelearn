#!/usr/bin/env bash
# Name: D.Saravanan
# Date: 30/07/2021
# Script to find number of files in the present working directory

number_of_files="$(find . * | wc -l)"

if [[ "$number_of_files" -gt 100 ]]; then
	echo "More than 100 files in the working directory!"
elif [[ "$number_of_files" -gt 10 ]]; then
	echo "More than 10 files in the working directory!"
else
	echo "Ten or fewer files in the working directory."
fi
