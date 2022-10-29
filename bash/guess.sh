#!/usr/bin/env bash
# File: guess.sh
# Name: D.Saravanan
# Date: 27/08/2020
# Bash script for number guessing game

option="yes"

while [ $option == "yes" ]
do
    declare -i N
	N=29

	echo -e "\\nI 'm thinking of a number! Try to guess the number I 'm thinking of: "
	read -r num

	while [ "$num" -ne $N ] 
	do
		if [ "$num" -lt $N ]
		then
			echo "Too low! Guess again: "
			read -r num

		else [ "$num" -gt $N ]
			echo "Too high! Guess again: "
			read -r num
		fi
	done

	if [ "$num" -eq $N ]
	then
		echo "That's it! Would you like to play again? (yes/no)"
		read -r option
	fi
done

echo -e "Thanks for playing!\\n"
