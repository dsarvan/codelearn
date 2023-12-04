/* File: serprog05.c
 * Name: D.Saravanan
 * Date: 04/12/2023
 * Program to initialize matrices, assign values and compute matrix multiplication
*/

#include <stdio.h>
#include <stdlib.h>

double **matrix(int nrow, int ncol) {

	double **data = (double **) calloc(nrow, sizeof(data));

	for (size_t i = 0; i < nrow; i++)
		data[i] = (double *) calloc(ncol, sizeof(data[i]));

	return data;
}

int main() {

	int nrow = 62, ncol = 15, nval = 7;

	double **A = matrix(nrow, ncol); /* matrix A */
	double **B = matrix(ncol, nval); /* matrix B */
	double **C = matrix(nrow, nval); /* matrix C */

	printf("Starting serial matrix multiplication example ...\n");
	printf("Using matrix sizes A[%d][%d], B[%d][%d], C[%d][%d]\n",
			nrow, ncol, ncol, nval, nrow, nval);

	printf("Initializing matrices ...\n");
	for (size_t i = 0; i < nrow; i++) /* Initialize matrix A */
		for (size_t j = 0; j < ncol; A[i][j] = i + j, j++);

	for (size_t i = 0; i < ncol; i++) /* Initialize matrix B */
		for (size_t j = 0; j < nval; B[i][j] = i * j, j++);

	/* matrix multiplication */
	printf("Performing matrix multiplication ...\n");
	for (size_t i = 0; i < nrow; i++) {
		for (size_t j = 0; j < nval; j++)
			for (size_t k = 0; k < ncol; C[i][j] += A[i][k] * B[k][j], k++);
	}

	free(A); free(B);

	/* print matrix result */
	printf("Here is the result matrix:");
	for (size_t i = 0; i < nrow; i++) {
		printf("\n");
		for (size_t j = 0; j < nval; j++)
			printf("%6.2e   ", C[i][j]);
	}

	free(C);
	printf("\n");

	return 0;
}
