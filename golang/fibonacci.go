// File: fibonacci.go
// Name: D.Saravanan
// Date: 15/10/2020
// Program to generate Fibonacci sequence

package main
import "fmt"

func fibonacci(x int) int {
	if x == 0 {
		return 0
	}
	else if x <= 2 {
		return 1
	}
	else {
		return fibonacci(x-2) + fibonacci(x-1)
	}
}

func main() {

	fmt.Println(fibonacci(5))

}
