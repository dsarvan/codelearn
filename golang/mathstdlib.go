// File: mathstdlib.go
// Name: D.Saravanan
// Date: 26/03/2024
// Program using math package in golang's standard library

package main

import (
    "fmt"
    "math"
)

func main() {

    // Mathematical constants
    // math constants like pi, e, phi
    // defined in the math package of
    // the standard library and they
    // have a precision till 15 digits
    // stored in float64 values

    fmt.Printf("Pi = %f\n", math.Pi)
    fmt.Printf("E = %f\n", math.E)
    fmt.Printf("Phi = %f\n", math.Phi)

    fmt.Printf("Sqrt2 = %f\n", math.Sqrt2)
    fmt.Printf("SqrtE = %f\n", math.SqrtE)
    fmt.Printf("SqrtPi = %f\n", math.SqrtPi)
    fmt.Printf("SqrtPhi = %f\n", math.SqrtPhi)

    fmt.Printf("Natural Log 2 = %f\n", math.Ln2)
    fmt.Printf("Natural Log 10 = %f\n", math.Ln10)
    fmt.Printf("1 / Natural Log 2 =  %f\n", math.Log2E)
    fmt.Printf("1 / Natural Log 10 = %f\n", math.Log10E)

}
