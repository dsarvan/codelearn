/* File: serprog06.c
 * Name: D.Saravanan
 * Date: 04/12/2023
 * Program to initialize matrices, assign values and compute matrix multiplication
*/

#include <stdio.h>
#include <stdlib.h>

int main() {

	int nrow = 10062, nval = 10015, ncol = 1007;

	double *A = (double *) calloc(nrow*nval, sizeof(*A)); /* matrix A */
	double *B = (double *) calloc(nval*ncol, sizeof(*B)); /* matrix B */
	double *C = (double *) calloc(nrow*ncol, sizeof(*C)); /* matrix C */

	printf("Starting serial matrix multiplication ...\n");
	printf("Using matrix sizes A[%d][%d], B[%d][%d], C[%d][%d]\n",
			nrow, nval, nval, ncol, nrow, ncol);

	printf("Initializing matrices ...\n");
	for (size_t i = 0; i < nrow; i++) /* Initialize matrix A */
		for (size_t j = 0; j < nval; A[i*nval + j] = i + j, j++);

	for (size_t i = 0; i < ncol; i++) /* Initialize matrix B */
		for (size_t j = 0; j < nval; B[i*nval + j] = i * j, j++);

	/* matrix multiplication */
	printf("Performing matrix multiplication ...\n");
	for (size_t i = 0; i < nrow; i++) {
		for (size_t j = 0; j < ncol; j++) {
			for (size_t k = 0; k < nval; k++)
				C[i*ncol + j] += A[i*nval + k] * B[k + nval*j];
		}
	}

	free(A); free(B);

	/* print matrix result */
	printf("Here is the result matrix:");
	for (size_t i = 0; i < nrow; i++) {
		printf("\n");
		for (size_t j = 0; j < ncol; j++)
			printf("%6.2e   ", C[i*ncol + j]);
	}
	printf("\n");

	free(C);
	return 0;
}
