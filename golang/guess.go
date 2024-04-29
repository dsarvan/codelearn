// File: guess.go
// Name: D.Saravanan
// Date: 23/02/2022
// Program for number guessing game

package main

import (
	"fmt"
	"math/rand"
	"time"
)

func main() {

	var option = "yes"
	var num int

	for option == "yes" {

		rand.Seed(time.Now().UnixNano())
		N := rand.Intn(100)

		fmt.Print("\nI 'm thinking of a number! Try to guess the number I 'm thinking of: ")
		fmt.Scanf("%d", &num)

		for num != N {

			if num < N {
				fmt.Print("Too low! Guess again: ")
				fmt.Scanf("%d", &num)
			} else if num > N {
				fmt.Print("Too high! Guess again: ")
				fmt.Scanf("%d", &num)
			}
		}

		if num == N {
			fmt.Print("That's it! Would you like to play again? (yes/no) ")
			fmt.Scanf("%s", &option)
		}

	}

	fmt.Println("Thanks for playing!")
}
