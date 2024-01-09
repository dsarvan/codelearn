#!/usr/bin/env bash
# File: age.sh
# Name: D.Saravanan
# Date: 24/02/2020
# Script to display number of persons in a given age group

echo " "

while read line; do
	echo "$line"
done <list.csv

min=$(cat list.csv | tail -n +2 | cut -d "," -f 3 | sort -n | head -n 1)
max=$(cat list.csv | tail -n +2 | cut -d "," -f 3 | sort -n | tail -n 1)

echo " "

echo "Enter lower limit age: "
read lower

echo " "

if [ $lower -lt $min ]; then
	echo "The minimum age value in the list is $min."
	echo "Enter lower limit age: "
	read lower
fi

echo "Enter upper limit age: "
read upper

if [ $upper -gt $max ]; then
	echo "The maximum age value in the list is $max."
	echo "Enter upper limit age: "
	read upper
fi

count=0

#No_of_entries=`cat list.csv | tail -n +2 | wc -l`
#echo $No_of_entries

#sorted_age_list=`cat list.csv | tail -n +2 | cut -d "," -f 3 | sort -n`
#echo $sorted_age_list

for i in {1..20}; do
	age[i]=$(cat list.csv | tail -n +2 | cut -d "," -f 3)

	if [ ${age[i]} -ge $lower ] && [ ${age[i]} -le $upper ]; then
		((count++))
	fi
done
