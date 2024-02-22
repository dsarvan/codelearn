#!/usr/bin/env bash
# File: pdfsize.sh
# Name: D.Saravanan
# Date: 22/02/2024
# Script to print the page size of a pdf file

# Usage: ./pdfsize.sh input.pdf

if [ -z "$(which pdfinfo)" ]; then
	echo "requires the pdfinfo utility provided by the poppler-utils package"
	exit 1
fi

# requires the pdfinfo utility provided by the poppler-utils package
pdfsize() {
	pdfinfo "$1" | grep 'Page size' |
		awk '{printf "Page size: %.2f x %.2f mm\n", $3*25.4/72, $5*25.4/72}'
}

# check if the parameter is a valid PDF file
if [[ -f "$1" && "$1" == *.pdf ]]; then
	pdfsize "$1"
else
	echo "Please provide a valid PDF file as a parameter."
fi
