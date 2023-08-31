// File: dynamfib.go
// Name: D.Saravanan
// Date: 29/11/2022
// Program to compute nth fibonacci using dynamic programming

package main

import "fmt"
import "math/big"

func fibonacci(n int) *big.Int {
	/* compute nth fibonacci */

	fib := [2]*big.Int{big.NewInt(0), big.NewInt(1)}

	for i := 0; i < n; i++ {
		fib[0].Add(fib[0], fib[1])
		fib[0], fib[1] = fib[1], fib[0]
	}

	return fib[0]
}

func main() {
	N := 10000
	fmt.Println(fibonacci(N))
}
