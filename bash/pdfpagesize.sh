#!/usr/bin/env bash
# File: pdfpagesize.sh
# Name: D.Saravanan
# Date: 22/02/2024
# Script takes a PDF file as a parameter and print the
# dimensions (width x height) in millimeters of each page

# Usage: ./pdfpagesize.sh input.pdf

# checks for the presence of the pdfinfo program on the system
if [ -z "$(which pdfinfo)" ]; then
	echo "requires the pdfinfo utility provided by the poppler-utils package"
	exit 1
fi

# check if the parameter is a valid PDF file
if [[ -f "$1" && "$1" == *.pdf ]]; then
	# loop through each page of the PDF file
	for page in $(seq 1 "$(pdfinfo "$1" | grep 'Pages' | awk '{print $2}')"); do

		# extract the width and height in points (1/72 inch) of the current page
		width=$(pdfinfo -f "$page" -l "$page" "$1" | grep "Page.*size" | awk '{print $4}')
		heigh=$(pdfinfo -f "$page" -l "$page" "$1" | grep "Page.*size" | awk '{print $6}')

		# convert the width and height to millimeters (25.4 mm = 1 inch)
		width_mm=$(bc -l <<<"$width * 25.4/72")
		heigh_mm=$(bc -l <<<"$heigh * 25.4/72")

		# print the dimensions of the current page
		printf "Page %d: %.2f x %.2f mm\n" "$page" "$width_mm" "$heigh_mm"
	done

else
	# print an error message if the parameter is not a valid PDF file
	echo "Please provide a valid PDF file as a parameter."
fi
