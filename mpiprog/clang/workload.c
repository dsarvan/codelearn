/* File: workload.c
 * Name: D.Saravanan
 * Date: 03/12/2023
 * Program to compute the workload of processes in a communicator
*/

#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {

	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	size_t N = 2000000000; /* array size */

	/* compute workload */
	int workload[size];
	for (size_t i = 0; i < size; workload[i] = N/size, i++);
	for (size_t i = 0; i < N%size; workload[i] += 1, i++);

	/* compute start load value and end load value */
	size_t sload = 0;
	for (size_t n = 0; n < rank; sload += workload[n], n++);
	size_t eload = sload + workload[rank];

	size_t n = eload - sload; /* number of elements in each process */

	/* print number of elements in each process with sload and eload */
	printf("rank %d of %d elements (%d, %d)\n", rank, n, sload, eload);

	MPI_Finalize();

	return 0;
}
