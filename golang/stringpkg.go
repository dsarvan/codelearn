// File: stringpkg.go
// Name: D.Saravanan
// Date: 07/05/2024
// Program using strings package

package main

import (
	"fmt"
	"strings"
)

func main() {

	// To search for a smaller string in a bigger string.
	// func Contains(s, substr string) bool
	fmt.Println(strings.Contains("test", "es"))

	// To count the number of times a smaller string occurs in a bigger string.
	// func Count(s, sep string) int
	fmt.Println(strings.Count("test", "t"))

	// To determine if a bigger string starts with a smaller string.
	// func HasPrefix(s, prefix string) bool
	fmt.Println(strings.HasPrefix("test", "te"))

	// To determine if a bigger string ends with a smaller string.
	// func HasSuffix(s, suffix string) bool
	fmt.Println(strings.HasSuffix("test", "st"))

	// To find the position of a smaller string in a bigger string.
	// func Index(s, sep string) int (it returns -1 if not found)
	fmt.Println(strings.Index("test", "e"))

	// To take a list of strings and join them together in a single string
	// separated by another string (e.g., a comma).
	// func Join(a []string, sep string) string
	fmt.Println(strings.Join([]string{"a", "b"}, "-"))

	// To repeat a string number of times.
	// func Repeat(s string, count int) string
	fmt.Println(strings.Repeat("a", 5))

	// To replace a smaller string in a bigger string with some other string.
	// func Replace(s, old, new string, n int) string
	fmt.Println(strings.Replace("aaaa", "a", "b", 2))

	// To split a string into a list of strings by a separating string.
	// func Split(s, sep string) []string
	fmt.Println(strings.Split("a-b-c-d-e", "-"))

	// To convert a string to all lowercase letters.
	// func ToLower(s string) string
	fmt.Println(strings.ToLower("TEST"))

	// To convert a string to all uppercase letters.
	// func ToUpper(s string) string
	fmt.Println(strings.ToUpper("test"))

	// To convert a string to a slice of bytes.
	arr := []byte("test")
	str := string([]byte{'t', 'e', 's', 't'})

}
