#!/usr/bin/env bash
# File: textanalysis.sh
# Name: D.Saravanan
# Date: 28/05/2021
# Script to analyse the text to count and sort the uniq words

less "$1" | tr ' ' '\n' | tr -d '[:punct:]' | tr '[:upper:]' '[:lower:]' | sort |
	uniq -c | sort -nr | less
