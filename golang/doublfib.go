// File: doublfib.go
// Name: D.Saravanan
// Date: 30/11/2022
// Program to compute nth fibonacci using doubling method

package main

import "fmt"
import "math/big"
import "math/bits"

func fibonacci(n int) *big.Int {
	/* compute nth fibonacci */

	nval := bits.Len64(uint64(n))
	x, y := big.NewInt(0), big.NewInt(0)

	fib := [4]*big.Int{big.NewInt(0), big.NewInt(1), big.NewInt(1), big.NewInt(2)}

	for m := nval - 1; m >= 0; m-- {

		fib[2] = big.NewInt(0).Mul(fib[0], y.Sub(x.Mul(big.NewInt(2), fib[1]), fib[0]))
		fib[3] = big.NewInt(0).Add(x.Mul(fib[0], fib[0]), y.Mul(fib[1], fib[1]))

		if (n>>m)&1 == 1 {
			fib[0], fib[1] = fib[3], big.NewInt(0).Add(fib[2], fib[3])
		} else {
			fib[0], fib[1] = fib[2], fib[3]
		}
	}

	return fib[0]
}

func main() {
	N := 10000
	fmt.Println(fibonacci(N))
}
