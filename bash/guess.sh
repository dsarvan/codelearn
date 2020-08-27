#!/usr/bin/env bash
# File: guess.sh
# Name: D.Saravanan
# Date: 27/08/2020
# Bash script for number guessing game

option="yes"

while [ $option == "yes" ]
do
	N=29

	echo -e "\nI 'm thinking of a number! Try to guess the number I 'm thinking of: "
	read num

	while [ $num -ne $N ] 
	do
		if [ $num -lt $N ]
		then
			echo "Too low! Guess again: "
			read num

		else [ $num -gt $N ]
			echo "Too high! Guess again: "
			read num
		fi
	done

	if [ $num -eq $N ]
	then
		echo "That's it! Would you like to play again? (yes/no)"
		read option
	fi
done

echo -e "Thanks for playing!\n"	
