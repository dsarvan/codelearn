// File: slices.go
// Name: D.Saravanan
// Date: 06/05/2024
// Program using slice data structure

package main

import "fmt"

func main() {

    n := 4
    t := make([]int, n) // slice

    t[0] = -1
    t[1] = -2
    t[2] = -3
    t[3] = -4

    fmt.Println(t)

    t = append(t, -5)

    fmt.Println(t)
}
