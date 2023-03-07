/* File: computeppi.c
 * Name: D.Saravanan
 * Date: 06/03/2023
 * Program to compute pi with mpi
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


double computepi(int rank, int size, signed N) {

	double dx = 1.0/N;

	int workloads[size];
	for (size_t i = 0; i < size; i++)
		workloads[i] = N/size;

	for (size_t i = 0; i < N%size; i++)
		workloads[i] += 1;

	size_t sload = 0;
	for (size_t n = 0; n < rank; n++)
		sload += workloads[n];
	size_t eload = sload + workloads[rank];

	double x(int i, double dx) {
		return (i + 0.5) * dx;
	}

	double lpi = 0.0;
	for (size_t i = sload; i < eload; i++)
		lpi = lpi + 4.0/(1 + x(i, dx)*x(i, dx)) * dx;

	double pi;
	MPI_Reduce(&lpi, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		return pi;
	}
}


int main(int argc, char **argv) {

	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	signed ns = 100000000;
	double pi = computepi(rank, size, ns);

	if (rank == 0) {
		printf("%f\n", pi);
	}

	MPI_Finalize();
	return 0;
}
