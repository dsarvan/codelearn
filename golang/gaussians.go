// File: gaussians.go
// Name: D.Saravanan
// Date: 25/06/2024
// Program to generate isolated Gaussian pulse

package main

import (
	"fmt"
	"math"
)

// Return evenly spaced numbers over a specified interval.
//
// :param float64 x1: the starting value of the sequence
// :param float64 x2: the end value of the sequence
// :param int n: number of samples to generate
//
// :return: n equally spaced samples in the close interval
func linspace(x1, x2 float64, n int) []float64 {
	x := make([]float64, n) // slice
	st := (x2 - x1) / float64(n-1)
	for i := 0; i < n; i++ {
		x[i] = x1 + float64(i)*st
	}
	return x
}

// Generate isolated Gaussian pulse
//
// :param int fs: sampling frequency in Hz
// :param float64 sigma: pulse width in seconds
//
// :return: (t, g): time base (t) and the signal g(t)
func gaussian(fs int, sigma float64) ([]float64, []float64) {

	g := make([]float64, fs) // slice

	t := linspace(-0.5, 0.5, fs)
	sval := 1 / (math.Sqrt(2*math.Pi) * sigma)

	for i := 0; i < fs; i++ {
		g[i] = sval * math.Exp(-0.5*math.Pow(t[i]/sigma, 2))
	}

	return t, g
}

func main() {

	fs := 80
	sigma := 0.1

	t, g := gaussian(fs, sigma)

	for i := 0; i < fs; i++ {
		fmt.Printf("%f, %f\n", t[i], g[i])
	}
}
