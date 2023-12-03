/* File: mpiprog04.c
 * Name: D.Saravanan
 * Date: 03/12/2023
 * Program to initialize an array, assign values,
 * demonstrates data decomposition and compute sum
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t N = 200000000; /* array size */
    double *data = (rank == 0) ? (double *) calloc(N, sizeof(*data)) : NULL;

    if (rank == 0) {
		printf("Starting parallel array example ...\n");
		printf("Using an array of %d floats requires %ld bytes\n", N, sizeof(*data));

		/* initializes array */
		printf("Initializing array ...\n");
		for (size_t i = 0; i < N; data[i] = i * 1.0, i++);
	}

    int workload[size];
    for (size_t i = 0; i < size; workload[i] = N / size, i++);
    for (size_t i = 0; i < N % size; workload[i] += 1, i++);

    size_t n = workload[rank]; /* array size */
    double *ndata = (double *) calloc(n, sizeof(*ndata));

    /* send data from process 0 to all other processes in a communicator */
    MPI_Scatter(data, n, MPI_DOUBLE, ndata, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* assign values and compute sum */
    printf("Performing computation on array elements ...\n");
    double nsum = 0;
    for (size_t i = 0; i < n; i++) {
		ndata[i] = ndata[i] + ndata[i];
		nsum += ndata[i];
    }

    /* receive data to process 0 from all other processes in a communicator */
    MPI_Gather(ndata, n, MPI_DOUBLE, data, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    free(ndata);

    /* collective computation sum operation */
    double sum = 0;
    MPI_Reduce(&nsum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* print sample results */
    if (rank == 0) {
		printf("Sample results:\n");
		for (size_t i = 10; i < N; i *= 10)
			printf("\t data[%d] = %e\n", i, data[i]);

		printf("\nFinal sum = %e\n", sum);
    }

    free(data);
    MPI_Finalize();

    return 0;
}
