/* File: matmult.c
 * Name: D.Saravanan
 * Date: 04/12/2023
 * Program to initialize matrices, assign values and compute matrix multiplication
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

double *matmult(int argc, char **argv, int nrow, int nval, int ncol) {

	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Status status;
	MPI_Request request;

	/* exit if nrow is not divisible by size */
	if (nrow % size != 0) {
		if (rank == 0)
			printf("Matrix size not divisible by process count %d.\n", size);
		MPI_Finalize();
		return 0;
	}

	int workload_A[size];
	for (size_t i = 0; i < size; workload_A[i] = nrow / size, i++);
	for (size_t i = 0; i < nrow % size; workload_A[i] += 1, i++);

	int workload_B[size];
	for (size_t i = 0; i < size; workload_B[i] = ncol / size, i++);
	for (size_t i = 0; i < ncol % size; workload_B[i] += 1, i++);

	size_t mrow = workload_A[rank]; /* no. of rows in matrix nA */
	size_t mcol = workload_B[rank]; /* no. of cols in matrix nB */

	size_t sload_A = 0;
	for (size_t n = 0; n < rank; sload_A += workload_A[n], n++);

	size_t sload_B = 0;
	for (size_t n = 0; n < rank; sload_B += workload_B[n], n++);

	if (rank == 0) {

		printf("Starting parallel matrix multiplication ...\n");
		printf("Using matrix sizes A[%d][%d], B[%d][%d], C[%d][%d]\n",
				nrow, nval, nval, ncol, nrow, ncol);

		printf("Initializing matrices ...\n");
	}

	double *nA = (double *) calloc(mrow*nval, sizeof(*nA)); /* matrix nA */

	for (size_t i = 0; i < mrow; i++) /* Initialize matrix A */
		for (size_t j = 0; j < nval; nA[i*nval + j] = (i + sload_A) + j, j++);

	double *nB = (double *) calloc(nval*mcol, sizeof(*nB)); /* matrix nB */

	for (size_t i = 0; i < mcol; i++) /* Initialize matrix B */
		for (size_t j = 0; j < nval; nB[i*nval + j] = (i + sload_B) * j, j++);

	double *nC = (double *) calloc(nrow*mcol, sizeof(*nC)); /* matrix nC */

	double sval = 0;
	int rval = rank;

	/* matrix multiplication */
	if (rank == 0) printf("Performing matrix multiplication ...\n");
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
		MPI_Isendrecv_replace(nA, mrow*nval, MPI_DOUBLE,
				  (rank == 0 ? size - 1 : rank - 1), 11,
				  (rank == size - 1 ? 0 : rank + 1), 11,
				  MPI_COMM_WORLD, &request);

		/* wait for all given communications to complete */
		MPI_Wait(&request, &status);
	}

	free(nA); free(nB);

	int *count_C = (int *) calloc(size, sizeof(*count_C)); /* no. of elements in nC */
	for (size_t i = 0; i < size; count_C[i] = nrow*workload_B[i], i++);

	int *displ_C = (int *) calloc(size, sizeof(*displ_C)); /* displacement data nC */
	for (size_t i = 1; i < size; displ_C[i] = displ_C[i - 1] + nrow*workload_B[i - 1], i++);

	double *M = (rank == 0) ? (double *) calloc(nrow*ncol, sizeof(*M)) : NULL; /* matrix M */

	/* gather matrix C elements to process 0 from all other process in communicator */
	MPI_Gatherv(nC, count_C[rank], MPI_DOUBLE, M, count_C, displ_C, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	free(nC); free(count_C); free(displ_C);

	double *C = (rank == 0) ? (double *) calloc(nrow*ncol, sizeof(*C)) : NULL; /* matrix C */

	/* matrix transpose */
	if (rank == 0) {
		for (size_t i = 0; i < ncol; i++) {
			for (size_t j = 0; j < nrow; j++)
				C[i + ncol*j] = M[i*nrow + j];
		}
	}

	free(M);

	if (rank == 0)
		return C;

	free(C);
	MPI_Finalize();
}
