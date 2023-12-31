/* File: sendrecv_replace.c
 * Name: D.Saravanan
 * Date: 28/11/2023
 * Program with blocking send/receive routines
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char **argv) {

    int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Status status;

    int nrow = 4, ncol = 7; /* number of rows and number of columns */

    if (nrow % size != 0) {
		if (rank == 0)
			printf("Matrix size not divisible by process count %d.\n", size);

		MPI_Finalize();
		return 0;
    }

    double *A = (rank == 0) ? (double *) calloc(nrow * ncol, sizeof(*A)) : NULL;

	/* initialize matrix elements if rank = 0 */
    if (rank == 0) {
		srand(time(NULL));
		for (size_t i = 0; i < nrow; i++)
			for (size_t j = 0; j < ncol; A[i * ncol + j] = rand() % 10, j++);
    }

	/* print matrix elements if rank = 0 */
    if (rank == 0) {
		printf("\n%d x %d Matrix A before scatter/gather:", nrow, ncol);
		for (size_t i = 0; i < nrow; i++) {
			printf("\n");
			for (size_t j = 0; j < ncol; j++) {
				printf("%0.2lf \t", A[i * ncol + j]);
			}
		}
		printf("\n");
    }

    int workload[size];
    for (size_t i = 0; i < size; workload[i] = nrow / size, i++);
    for (size_t i = 0; i < nrow % size; workload[i] += 1, i++);

    size_t mrow = workload[rank]; /* number of rows send to each process */
    double *nA = (double *) calloc(mrow * ncol, sizeof(*nA));

	/* integer array specifying the number of elements to send to each process */
    int *count = (int *) calloc(size, sizeof(*count));
    for (size_t i = 0; i < size; count[i] = workload[i] * ncol, i++);

	/* integer array specifying the displacement (relative to sendbuf) */
    int *displ = (int *) calloc(size, sizeof(*displ));
    for (size_t i = 1; i < size; displ[i] = displ[i - 1] + workload[i - 1]*ncol, i++);

	/* print count and displ information */
	if (rank == 0) {
		printf("\ncount and displ information:\n");
		for (size_t i = 0; i < size; i++) {
			printf("count[%d] = %d, \t displ[%d] = %d\n", i, count[i], i, displ[i]);
		}
	}

	/* send variable data from process 0 to all other processes in a communicator */
    MPI_Scatterv(A, count, displ, MPI_DOUBLE, nA, count[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	/* print scatter matrix elements if rank = 1 */
	if (rank == 1) {
		printf("\n%d x %d Matrix nA of rank %d:", mrow, ncol, rank);
		for (size_t i = 0; i < mrow; i++) {
			printf("\n");
			for (size_t j = 0; j < ncol; j++) {
				printf("%0.2lf \t", nA[i * ncol + j]);
			}
		}
		printf("\n");
	}

    for (size_t iter = 0; iter < size - 1; iter++) {

		//MPI_Send(nA, mrow*ncol, MPI_DOUBLE, (rank == 0 ? size - 1 : rank - 1), 11,
		//	MPI_COMM_WORLD);
		//MPI_Recv(nA, mrow*ncol, MPI_DOUBLE, (rank == size - 1 ? 0 : rank + 1), 11,
		//	MPI_COMM_WORLD, &status);

		MPI_Sendrecv_replace(nA, mrow*ncol, MPI_DOUBLE, (rank == 0 ? size - 1 : rank - 1), 11,
					        (rank == size - 1 ? 0 : rank + 1), 11, MPI_COMM_WORLD, &status);

    }

	/* receive variable data to process 0 from all other processes in a communicator */
    MPI_Gatherv(nA, count[rank], MPI_DOUBLE, A, count, displ, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(count); free(displ); free(nA);

	/* print gather matrix elements if rank = 0 */
    if (rank == 0) {
		printf("\n%d x %d Matrix A after scatter/gather:", nrow, ncol);
		for (size_t i = 0; i < nrow; i++) {
			printf("\n");
			for (size_t j = 0; j < ncol; j++) {
				printf("%0.2lf \t", A[i * ncol + j]);
			}
		}
		printf("\n");
    }

    free(A);
    MPI_Finalize();

    return 0;
}
