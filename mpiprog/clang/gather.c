/* File: gather.c
 * Name: D.Saravanan
 * Date: 26/11/2023
 * Program that receives data to process 0 from all other processes in a communicator
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char **argv) {

	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	double time = MPI_Wtime();

	int N = 200; /* array size */
	double *data = (rank == 0) ? (double *) calloc(N, sizeof(data)) : NULL;

	if (rank == 0) /* initialize array if rank = 0 */
		for (size_t i = 0; i < N; data[i] = i * 1.0, i++);

	int workload[size];
	for (size_t i = 0; i < size; workload[i] = N / size, i++);
	for (size_t i = 0; i < N % size; workload[i] += 1, i++);

	size_t n = workload[rank]; /* number of elements sent to each process */
	double *ndata = (double *) calloc(n, sizeof(*ndata));

	/* send data from process 0 to all other processes in a communicator */
	MPI_Scatter(data, n, MPI_DOUBLE, ndata, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	/* assign values */
	for (size_t i = 0; i < n; ndata[i] = 2 * ndata[i], i++);

	printf("Scatter data (process = %d):\n", rank);
	for (size_t i = n - 5; i < n; i++)
		printf("rank = %d   data[%d] = %0.1lf\n", rank, i, ndata[i]);

	/* receive data to process 0 from all other processes in a communicator */
	MPI_Gather(ndata, n, MPI_DOUBLE, data, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	free(ndata);

	if (rank == 0) {
	printf("Gather data (process = 0):\n");
	for (size_t i = N - 5; i < N; i++)
		printf("rank = %d   data[%d] = %0.1lf\n", rank, i, data[i]);
	}

	free(data);

	time = MPI_Wtime() - time;

	printf("Timing from rank %d is %lf seconds\n", rank, time);

	MPI_Finalize();
	return 0;
}
