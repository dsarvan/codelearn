// File: mathstdlib.go
// Name: D.Saravanan
// Date: 26/03/2024
// Program using math package in golang's standard library

package main

import (
	"fmt"
	"math"
	"math/bits"
	"math/cmplx"
	"math/rand"
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

	// Basic Math functions

	// Abs: parameters(float64), returns float64
	fmt.Printf("Absolute value = %f\n", math.Abs(float64(45)-float64(100)))

	// Type casting
	num := 77
	fmt.Printf("Type of num = %T\n", num)
	fmt.Printf("num integer cast = %d\n", int(num))
	fmt.Printf("num float cast = %f\n", float64(num))
	fmt.Printf("num string cast = %s\n", string(num))

	fmt.Printf("bool cast true = %t\n", bool(true))
	fmt.Printf("bool cast false = %t\n", bool(false))

	// Note: the math package almost deals with float64 types rather
	// than int to avoid backwards compatibility to perform operations
	// on floating point values which can be casted into integers rather
	// than defining separate functions for decimal values and integers

	// Min/Max: parameters(float64), returns float64
	var x1 float64 = 54
	var x2 float64 = 120

	minimum := math.Min(float64(x1), float64(x2))
	maximum := math.Max(float64(x1), float64(x2))

	fmt.Printf("Minimum of %v and %v is %v\n", x1, x2, minimum)
	fmt.Printf("Maximum of %v and %v is %v\n", x1, x2, maximum)

	var x float64 = 3
	var y float64 = 4

	// Pow: parameters(float64, float64), returns float64
	fmt.Printf("Exponential result %v**%v = %v\n", x, y, math.Pow(x, y))

	// Pow10: parameters(int), returns float64
	fmt.Printf("Exponential result 10**%v = %v\n", x, math.Pow10(int(x)))

	// Sqrt: parameters(float64), returns float64
	fmt.Printf("Square root of %v = %v\n", y, math.Sqrt(y))

	// Cbrt: parameters(float64), returns float64
	fmt.Printf("Cube root of %v = %v\n", y, math.Cbrt(y))

	// Trunc: parameters(float64), returns float64
	// Truncate function provides the way to round off a decimal value (float64)
	// to an integer but it returns a value in float64
	var result float64 = 445.235
	fmt.Printf("Truncated value of %v = %v\n", result, math.Trunc(result))
	result = 123.678
	fmt.Printf("Truncated value of %v = %v\n", result, math.Trunc(result))

	// Ceil: parameters(float64), returns float64
	// Ceil function to round up the value to the next integer value
	// but the value is returned as float64
	result = 33.25
	fmt.Printf("Ceiled value of %v = %v\n", result, math.Ceil(result))
	result = 134.78
	fmt.Printf("Ceiled value of %v = %v\n", result, math.Ceil(result))

	// Trigonometric functions: parametets(float64), returns float64
	// basic trigonometric functions: sin, cos, tan
	fmt.Printf("sin(%v) = %v\n", math.Pi/2, math.Sin(math.Pi/2))
	fmt.Printf("cos(%v) = %v\n", math.Pi/2, math.Cos(math.Pi/2))
	fmt.Printf("tan(%v) = %v\n", math.Pi/2, math.Tan(math.Pi/2))
	// hyperbolic trigonometric functions: sinh, cosh, tanh
	fmt.Printf("sinh(%v) = %v\n", math.Pi/2, math.Sinh(math.Pi/2))
	fmt.Printf("cosh(%v) = %v\n", math.Pi/2, math.Cosh(math.Pi/2))
	fmt.Printf("tanh(%v) = %v\n", math.Pi/2, math.Tanh(math.Pi/2))
	// inverse trigonometric functions: asin, acos, atan
	fmt.Printf("asin(%v) = %v\n", math.Pi/8, math.Asin(math.Pi/8))
	fmt.Printf("acos(%v) = %v\n", math.Pi/8, math.Acos(math.Pi/8))
	fmt.Printf("atan(%v) = %v\n", math.Pi/8, math.Atan(math.Pi/8))
	// inverse hyperbolic trigonometric functions: asinh, acosh, atanh
	fmt.Printf("asinh(%v) = %v\n", 2*math.Pi, math.Asinh(2*math.Pi))
	fmt.Printf("acosh(%v) = %v\n", 2*math.Pi, math.Acosh(2*math.Pi))
	fmt.Printf("atanh(%v) = %v\n", 2*math.Pi, math.Atanh(2*math.Pi))

	// Exponential functions
	// Exp: parameters(float64), returns float64
	fmt.Printf("exp(%v) = %v\n", x1, math.Exp(x1))
	// Exp2: parameters(float64), returns float64
	fmt.Printf("exp2(%v) = %v\n", x2, math.Exp2(x2))

	// Logarithmic functions
	// Log: parameters(float64), returns float64
	fmt.Printf("log(%v) = %v\n", x1, math.Log(x1))
	// Log2: parameters(float64), returns float64
	fmt.Printf("log2(%v) = %v\n", x2, math.Log2(x2))

	// Random package
	// Int: parameters(), returns int
	fmt.Printf("pseudo random integer generation, x = %v\n", rand.Int())
	// Intn: parameters(int), returns int
	for i := 0; i < 5; i++ {
		fmt.Printf("pseudo random number generation range, x = %v\n", rand.Intn(100))
	}

	// Bits package
	// Add: parameters(uint, uint, uint), returns (uint, uint)
	sum, carry := bits.Add(0, 9, 1)
	fmt.Printf("Sum = %d, Carry = %d\n", sum, carry)
	// Len: parameters(uint), returns int
	var n uint = 45 // (45) in decimal = (1 0 1 1 0 1) in binary
	fmt.Printf("Minimum bits required to represent %v = %d\n", n, bits.Len(n))
	// OnesCount: parameters(uint), returns int
	fmt.Printf("Set Bits in %v = %d\n", n, bits.OnesCount(n))

	// Complex package
	xc := complex(5, 8)
	yc := complex(3, 4)

	fmt.Printf("Complex number x = %v\n", xc)
	fmt.Printf("Complex number y = %v\n", yc)
	fmt.Printf("Modulus of x = %v\n", cmplx.Abs(xc))
	fmt.Printf("Modulus of y = %v\n", cmplx.Abs(yc))
	fmt.Printf("Conjugate of x = %v\n", cmplx.Conj(xc))
	fmt.Printf("Conjugate of y = %v\n", cmplx.Conj(yc))
	fmt.Printf("Phase of x = %v\n", cmplx.Phase(xc))
	fmt.Printf("Phase of y = %v\n", cmplx.Phase(yc))

	modx, phasex := cmplx.Polar(xc)
	mody, phasey := cmplx.Polar(yc)
	fmt.Printf("Polar form of x = %v, %v\n", modx, phasex)
	fmt.Printf("Polar form of y = %v, %v\n", mody, phasey)

}
