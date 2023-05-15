// File: statistics.go
// Name: D.Saravanan
// Date: 15/05/2023
// Program to calculate the mean, median, standard-dev
// and variance of a small dataset using gonum package

package main

import (
	"fmt"
	"math"
	"sort"

	"gonum.org/v1/gonum/stat"
)

func main() {

	xs := []float64{
		32.32, 56.98, 21.52, 44.32, 55.63,
		13.75, 43.47, 43.34, 12.34, 35.72,
	}

	fmt.Printf("data: %v\n", xs)

	// computes the weighted mean of the dataset
	// all weights are 1 (so we pass a nil slice)
	mean := stat.Mean(xs, nil)
	variance := stat.Variance(xs, nil)
	stddev := math.Sqrt(variance)

	// stat.Quantile needs the input slice to be sorted
	sort.Float64s(xs)
	fmt.Printf("data: %v (sorted)\n", xs)

	// computes the median of the dataset
	// all weights are 1 (so we pass a nil slice)
	median := stat.Quantile(0.5, stat.Empricial, xs, nil)

	fmt.Printf("mean = %v\n", mean)
	fmt.Printf("median = %v\n", median)
	fmt.Printf("variance = %v\n", variance)
	fmt.Printf("std-dev = %v\n", stddev)
}
