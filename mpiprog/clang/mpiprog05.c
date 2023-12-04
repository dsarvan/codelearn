/* File: mpiprog05.c
 * Name: D.Saravanan
 * Date: 04/12/2023
 * Program to initialize matrices, assign values, demonstrates
 * data decomposition and compute matrix multiplication
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {

	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Status status;
	MPI_Request request;

	int nrow = 62, nval = 15, ncol = 7;

	/* exit if nrow is not divisible by size */
	if (nrow % size != 0) {
		if (rank == 0)
			printf("Matrix size not divisible by process count %d.\n", size);
		MPI_Finalize();
		return 0;
	}

	double *A = (rank == 0) ? (double *) calloc(nrow*nval, sizeof(*A)) : NULL; /* matrix A */
	double *B = (rank == 0) ? (double *) calloc(nval*ncol, sizeof(*B)) : NULL; /* matrix B */
	double *C = (rank == 0) ? (double *) calloc(nrow*ncol, sizeof(*C)) : NULL; /* matrix C */

	if (rank == 0) {

		printf("Starting parallel matrix multiplication example ...\n");
		printf("Using matrix sizes A[%d][%d], B[%d][%d], C[%d][%d]\n",
				nrow, nval, nval, ncol, nrow, ncol);

		printf("Initializing matrices ...\n");
		for (size_t i = 0; i < nrow; i++) /* Initialize matrix A */
			for (size_t j = 0; j < nval; A[i*nval + j] = i + j, j++);

		for (size_t i = 0; i < ncol; i++) /* Initialize matrix B */
			for (size_t j = 0; j < nval; B[i*nval + j] = i * j, j++);
	}

	int workload_A[size];
	for (size_t i = 0; i < size; workload_A[i] = nrow / size, i++);
	for (size_t i = 0; i < nrow % size; workload_A[i] += 1, i++);

	int workload_B[size];
	for (size_t i = 0; i < size; workload_B[i] = ncol / size, i++);
	for (size_t i = 0; i < ncol % size; workload_B[i] += 1, i++);

	size_t mrow = workload_A[rank]; /* no. of rows in matrix nA */
	size_t mcol = workload_B[rank]; /* no. of cols in matrix nB */

	double *nA = (double *) calloc(mrow*nval, sizeof(*nA)); /* matrix nA */
	double *nB = (double *) calloc(nval*mcol, sizeof(*nB)); /* matrix nB */
	double *nC = (double *) calloc(nrow*mcol, sizeof(*nC)); /* matrix nC */

	int *count_A = (int *) calloc(size, sizeof(*count_A)); /* no. of elements in nA */
	for (size_t i = 0; i < size; count_A[i] = workload_A[i]*nval, i++);

	int *displ_A = (int *) calloc(size, sizeof(*displ_A)); /* displacement data nA */
	for (size_t i = 1; i < size; displ_A[i] = displ_A[i - 1] + workload_A[i - 1]*nval, i++);

	/* scatter matrix A elements from process 0 to all other process in communicator */
	MPI_Scatterv(A, count_A, displ_A, MPI_DOUBLE, nA, count_A[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	free(A); free(count_A); free(displ_A);

	int *count_B = (int *) calloc(size, sizeof(*count_B)); /* no. of elements in nB */
	for (size_t i = 0; i < size; count_B[i] = nval*workload_B[i], i++);

	int *displ_B = (int *) calloc(size, sizeof(*displ_B)); /* displacement data nB */
	for (size_t i = 1; i < size; displ_B[i] = displ_B[i - 1] + nval*workload_B[i - 1], i++);

	/* scatter matrix B elements from process 0 to all other process in communicator */
	MPI_Scatterv(B, count_B, displ_B, MPI_DOUBLE, nB, count_B[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	free(B); free(count_B); free(displ_B);

	int *count_C = (int *) calloc(size, sizeof(*count_C)); /* no. of elements in nC */
	for (size_t i = 0; i < size; count_C[i] = nrow*workload_B[i], i++);

	int *displ_C = (int *) calloc(size, sizeof(*displ_C)); /* displacement data nC */
	for (size_t i = 1; i < size; displ_C[i] = displ_C[i - 1] + nrow*workload_B[i - 1], i++);

	/* scatter matrix C elements from process 0 to all other process in communicator */
	MPI_Scatterv(C, count_C, displ_C, MPI_DOUBLE, nC, count_C[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double sval = 0;
	int rval = rank;

	/* matrix multiplication */
	printf("Performing matrix multiplication ...\n");
	for (size_t iter = 0; iter < size; iter++) {

		for (size_t i = 0; i < mrow; i++) {
			for (size_t j = 0; j < mcol; j++) {

				for (size_t k = 0; k < nval; k++)
					sval += nA[i*nval + k] * nB[k + nval*j];

				nC[i + rval*nrow/size + nrow*j] = sval;
				sval = 0;
			}
		}

		rval = (rval == size - 1) ? 0 : rval + 1;

		/* non-blocking send and receive routines */
		MPI_Isend(nA, mrow*nval, MPI_DOUBLE, (rank == 0 ? size - 1 : rank - 1), 11, MPI_COMM_WORLD, &request);
		MPI_Irecv(nA, mrow*nval, MPI_DOUBLE, (rank == size - 1 ? 0 : rank + 1), 11, MPI_COMM_WORLD, &request);
		MPI_Wait(&request, &status); /* wait for all given communications to complete */
	}

	free(nA); free(nB);

	double *M = (rank == 0) ? (double *) calloc(nrow*ncol, sizeof(*M)) : NULL; /* matrix M */

	/* gather matrix C elements to process 0 from all other process in communicator */
	MPI_Gatherv(nC, count_C[rank], MPI_DOUBLE, M, count_C, displ_C, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	free(nC); free(count_C); free(displ_C);

	/* matrix transpose */
	if (rank == 0) {
		for (size_t i = 0; i < ncol; i++) {
			for (size_t j = 0; j < nrow; j++)
				C[i + ncol*j] = M[i*nrow + j];
		}
	}

	free(M);

	/* print matrix result */
	if (rank == 0) {
		printf("Here is the result matrix:");
		for (size_t i = 0; i < nrow; i++) {
			printf("\n");
			for (size_t j = 0; j < ncol; j++)
				printf("%6.2e   ", C[i*ncol + j]);
		}
		printf("\n");
	}

	free(C);
	MPI_Finalize();

	return 0;
}
