// File: perfectnumber.go
// Name: D.Saravanan
// Date: 13/05/2023
// Program to find a perfect number

package main

import "fmt"

func main() {

	const n = 10000

	for x := 2; x <= n; x++ {
		var sum = 0
		for n := 1; n <= x/2; n++ {
			if x%n == 0 {
				sum += n
			}
		}
		if x == sum {
			fmt.Printf("%d\n", x)
		}
	}
}
