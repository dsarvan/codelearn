#!/usr/bin/env bash
# File: empscript.sh
# Name: D.Saravanan
# Date: 24/02/2020
# Script to read user input and write/append to a file

# create a file empfile.txt
touch empfile.txt

# write header (EmpID, EmpName, Basic) to empfile.txt
#echo "EmpID | EmpName | Basic" > empfile.txt
printf "%4s\t%20s\t%8s\n" "EmpID | EmpName | Basic" >empfile.txt

# ask how many number of records to be entered
echo "Enter number of records: "
read n

i=1       # initialize i = 1
empid=100 # initialize empid = 100

# start of while loop
while [ $i -le $n ]; do
	((empid++)) # auto increment empid by 1
	echo "Enter name: "
	read empname
	echo "Enter basic: "
	read basic

	#echo "$empid | $empname | $basic" >> empfile.txt
	printf "%4s\t%20s\t%8s\n" "$empid | $empname | $basic" >>empfile.txt

	((i++)) # increment i by 1
done
