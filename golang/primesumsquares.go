// File: primesumsquares.go
// Name: D.Saravanan
// Date: 14/05/2023
// Program to check an odd prime number has sum of two squares

package main

import (
	"fmt"
	"math"
)

func sumsquares(nval int) int {
	// An odd prime number p, the sum of
	// two squares if and only if it leaves
	// the remainder 1 on division by 4.
	if nval%4 == 1 {
		fmt.Printf("%d\n", nval)
	}

	return 0
}

func prime(number int) int {
	// function to check prime
	numval := float64(number)
	var sqrt_number float64 = math.Sqrt(numval)
	for i := 2; i <= int(sqrt_number); i++ {
		if number%i == 0 {
			return 0
		}
	}
	return sumsquares(number)
}

func main() {
	const N = 1000
	for n := 2; n <= N; n++ {
		prime(n)
	}
}
