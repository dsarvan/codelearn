// File: surface.go
// Name: D.Saravanan
// Date: 15/05/2023
// Program to compute an svg rendering of 3 dimensional surface function

package main

import (
	"fmt"
	"math"
)

const (
	width, height = 600, 320            // canvas size in pixels
	cells         = 100                 // number of grid cells
	xyrange       = 30.0                // axis ranges (-xyrange...+xyrange)
	xyscale       = width / 2 / xyrange // pixels per x or y unit
	zscale        = height * 0.4        // pixels per z unit
	angle         = math.Pi / 6         // angle of x, y axes
)

func f(x, y float64) float64 {
	r := math.Hypot(x, y) // distance from (0, 0)
	return math.Sin(r) / r
}

func corner(i, j int) (float64, float64) {
	// Find point (x, y) at corner of cell (i, j)
	x := xyrange * (float64(i)/cells - 0.5)
	y := xyrange * (float64(j)/cells - 0.5)

	// Compute surface height z
	z := f(x, y)

	// Project (x, y, z) isometrically onto 2-D SVG canvas(sx, sy)
	sx := width/2 + (x-y)*math.Cos(angle)*xyscale
	sy := height/2 + (x+y)*math.Sin(angle) - z*zscale
	return sx, sy
}

func main() {

	for i := 0; i <= cells; i++ {
		for j := 0; j <= cells; j++ {
			ax, ay := corner(i+1, j)
			bx, by := corner(i, j)
			cx, cy := corner(i, j+1)
			dx, dy := corner(i+1, j+1)
		}
	}
}
