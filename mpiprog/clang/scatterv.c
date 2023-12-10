/* File: scatterv.c
 * Name: D.Saravanan
 * Date: 27/11/2023
 * Program that sends variable number of data from one process to all other processes in a communicator
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {

	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int nrow = 11, ncol = 6; /* number of rows and number of columns */
	double *data = (rank == 0) ? (double *) calloc(nrow*ncol, sizeof(*data)) : NULL;

	/* initialize matrix elements if rank = 0 */
	int ncount = 0;
	if (rank == 0) {
		for (size_t i = 0; i < nrow; i++) {
			for (size_t j = 0; j < ncol; j++) {
				data[i*ncol + j] =  1.0 * ncount;
				ncount += 1;
			}
		}
	}

	/* print matrix elements if rank = 0 */
	if (rank == 0) {
		printf("%d x %d Matrix:", nrow, ncol);
		for (size_t i = 0; i < nrow; i++) {
			printf("\n");
			for (size_t j = 0; j < ncol; j++) {
				printf("[%d] %0.1lf \t", rank, data[i*ncol + j]);
			}
		}
		printf("\n");
	}

	int workload[size];
	for (size_t i = 0; i < size; workload[i] = nrow / size, i++);
	for (size_t i = 0; i < nrow % size; workload[i] += 1, i++);

	size_t mrow = workload[rank]; /* number of rows sent to each process */
	double *ndata = (double *) calloc(mrow*ncol, sizeof(*ndata));

	/* integer array specifying the number of elements to send to each process */
	int *count = (int *) calloc(size, sizeof(*count));
	for (size_t i = 0; i < size; count[i] = workload[i]*ncol, i++);

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
	MPI_Scatterv(data, count, displ, MPI_DOUBLE, ndata, count[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	free(data); free(count); free(displ);

	/* print scatter matrix elements if rank = 2 */
	if (rank == 2) {
		printf("\n%d x %d Matrix:", mrow, ncol);
		for (size_t i = 0; i < mrow; i++) {
			printf("\n");
			for (size_t j = 0; j < ncol; j++) {
				printf("[%d] %0.1lf \t", rank, ndata[i*ncol + j]);
			}
		}
		printf("\n");
	}

	free(ndata);

	MPI_Finalize();
	return 0;
}
