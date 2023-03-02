#!/usr/bin/env bash
# File: list.sh
# Name: D.Saravanan    
# Date: 24/02/2020
# Script to read input from user and write/append to a file

touch list.csv

printf "%4s\t%20s\t%3s\t%6s\n" "UniqueID, Name, Age, Pincode" > list.csv

echo "Enter number of records: "
read n

i=1
uniqueid=100

while [ $i -le $n ]
do
	((uniqueid++))
	echo "Enter name: "
	read name
	echo "Enter age: "
	read age
	echo "Enter pincode: "
	read pincode

	printf "%4s\t%20s\t%3s\t%6s\n" "$uniqueid, $name, $age, $pincode" >> list.csv

	((i++))
done
