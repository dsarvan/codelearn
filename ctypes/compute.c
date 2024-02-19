/* File: compute.c
 * Name: D.Saravanan
 * Date: 19/02/2024
 * Program to compute array elements sum
*/

#include <stdio.h>

double sumval(int n, double *arr) {
	/* summation */
	double sum = 0;
	for (int i = 0; i < n; sum += arr[i], i++);
	return sum;
}
