// File: parallel.c
// Name: D.Saravanan
// Date: 11/11/2022

/* MPI Sendrecv to exchange ghost cell values */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int localsize(int rank, int size, int N);

int main(int argc, char **argv) {

	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int lproc = rank - 1;
	if (lproc < 0) lproc = MPI_PROC_NULL;
	int rproc = rank + 1;
	if (rproc >= size) rproc = MPI_PROC_NULL;

	int N = 10;

	int n = localsize(rank, size, N);

	int *ex = (int *) malloc((n+2) * sizeof(int));

	for (int i=1; i<=n; i++) {
		ex[i] = 123;
		printf("%d\t%d\n", rank, i);
	}

	// ghost cells
	ex[0] = 8; ex[n+1] = 8;

	// sending ghost value to the right
	MPI_Sendrecv(&ex[n], 1, MPI_INT, rproc, 12345, &ex[0], 1, MPI_INT, lproc, 12345, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	// sending ghost value to the left
	MPI_Sendrecv(&ex[1], 1, MPI_INT, lproc, 12345, &ex[n+1], 1, MPI_INT, rproc, 12345, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	/* mpiexec -np 2 ./parallel */
	printf("[%d] The head ghost is: %d\n", rank, ex[0]);
	printf("[%d] The tail ghost is: %d\n", rank, ex[n+1]);
	
	free(ex);

	MPI_Finalize();

	return 0;
}


int localsize(int rank, int size, int N) {

	int n = N/size;

	double eval = N - n*size;

	if (rank < eval) n = n + 1;

	return n;
}
