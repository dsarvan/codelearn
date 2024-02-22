#!/usr/bin/env bash
# File: pdfsize.sh
# Name: D.Saravanan
# Date: 22/02/2024
# Script to print the page size of a pdf file

# requires the pdfinfo utility provided by the poppler-utils package
pdfsize() {
	pdfinfo "$1" | grep 'Page size' |
		awk '{printf "Page size: %.2f x %.2f mm\n", $3*25.4/72, $5*25.4/72}'
}

pdfsize "$1"
