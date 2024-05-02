// File: fibonacci.go
// Name: D.Saravanan
// Date: 15/10/2020
// Program to generate Fibonacci sequence

package main
import "fmt"

func fibonacci(n int) int {
    x, y := 0, 1

    for i := 0; i < n; i++ {
        x, y = y, x+y
    }

    return x
}

func main() {

	fmt.Println(fibonacci(100))

}
