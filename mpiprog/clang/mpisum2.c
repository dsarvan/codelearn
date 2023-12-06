/* File: mpisum2.c
 * Name: D.Saravanan
 * Date: 03/12/2023
 * Program to initialize an array on each processes in a communicator,
 * assign values, compute sum on each process and compute collective sum operation
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	double time = MPI_Wtime();

    int N = 200000000; /* array size */

    int workload[size];
    for (size_t i = 0; i < size; workload[i] = N / size, i++);
    for (size_t i = 0; i < N % size; workload[i] += 1, i++);

	size_t sload = 0;
	for (size_t n = 0; n < rank; sload += workload[n], n++);

    size_t n = workload[rank]; /* number of elements sent to each process */
    double *ndata = (double *) calloc(n, sizeof(*ndata));

	/* initialize array on each process */
	for (size_t i = 0; i < n; ndata[i] = (i + sload) * 1.0, i++);

	/* assign values and compute sum */
	double nsum = 0;
    for (size_t i = 0; i < n; i++) {
		ndata[i] = ndata[i] + ndata[i];
		nsum += ndata[i];
	}

    free(ndata);

	/* collective computation sum operation */
	double sum = 0;
	MPI_Reduce(&nsum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0) /* print result */
		printf("Final sum = %e\n", sum);

	time = MPI_Wtime() - time;
	printf("Timing from rank %d is %lf seconds\n", rank, time);

    MPI_Finalize();

    return 0;
}
