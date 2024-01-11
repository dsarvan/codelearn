#!/usr/bin/env bash
# File: combination.sh
# Name: D.Saravanan
# Date: 05/01/2021
# Script to calculate combination C(n,r)

factorial() {

	declare -i value=1

	for ((x = $1; x >= 1; x--)); do
		value=$((value * x))
	done

	return $value
}

declare -i n
declare -i r

printf "Enter n: "
read -r n
printf "Enter r: "
read -r r

factorial "$n"
value1=$?
factorial "$n"-"$r"
value2=$?
factorial "$r"
value3=$?

((combination = value1 / (value2 * value3)))
printf "Combination C($n,$r): %d\\n" $combination
