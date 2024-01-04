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

	int nrow = 10062, nval = 10015, ncol = 1007;

	double **A = matrix(nrow, nval); /* matrix A */
	double **B = matrix(nval, ncol); /* matrix B */
	double **C = matrix(nrow, ncol); /* matrix C */

	printf("Starting serial matrix multiplication ...\n");
	printf("Using matrix sizes A[%d][%d], B[%d][%d], C[%d][%d]\n",
			nrow, nval, nval, ncol, nrow, ncol);

	printf("Initializing matrices ...\n");
	for (size_t i = 0; i < nrow; i++) /* Initialize matrix A */
		for (size_t j = 0; j < nval; A[i][j] = i + j, j++);

	for (size_t i = 0; i < nval; i++) /* Initialize matrix B */
		for (size_t j = 0; j < ncol; B[i][j] = i * j, j++);

	/* matrix multiplication */
	printf("Performing matrix multiplication ...\n");
	for (size_t i = 0; i < nrow; i++) {
		for (size_t j = 0; j < ncol; j++)
			for (size_t k = 0; k < nval; C[i][j] += A[i][k] * B[k][j], k++);
	}

	free(A); free(B);

	/* print matrix result */
	printf("Here is the result matrix:");
	for (size_t i = 0; i < nrow; i++) {
		printf("\n");
		for (size_t j = 0; j < ncol; j++)
			printf("%6.2e   ", C[i][j]);
	}
	printf("\n");

	free(C);
	return 0;
}
