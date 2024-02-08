#!/usr/bin/env bash
# File: djvutk.sh
# Name: D.Saravanan
# Date: 07/02/2024
# Script to remove pages from input file to create output file

# Usage:
# ./djvutk.sh input.djvu start_page end_page/end output.djvu

spage=$2 # start page number
epage=$3 # end page number

input_file=$1  # input file name
output_file=$4 # output file name

if [ $# == 4 ]; then

	# copy input file to output file
	cp "$input_file" "$output_file"

	# number of pages in output file
	end=$(djvused -e 'n' "$output_file")

	for ((i = 1; i < spage; i++)); do
		djvm -delete "$output_file" 1
	done

	for ((i = end; i > epage; i--)); do
		djvm -delete "$output_file" $i
	done

else
	echo "Oops, there was an error!"
	echo "Usage: ./djvutk.sh input.djvu start_page end_page/end output.djvu"

fi
