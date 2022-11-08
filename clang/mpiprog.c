/* File: mpiprog.c
 * Name: D.Saravanan
 * Date: 08/11/2022
 * MPI program working example
*/

#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {

	int rank, size;

	/* Initializes after program start, 
	including program arguments */
	MPI_Init(&argc, &argv);

	/* Gets the rank number of the process */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/* Gets the number of ranks in the program 
	determined by the mpiexec command */
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	printf("rank %d of nproc %d\n", rank, size);

	/* Finalizes MPI to synchronize ranks and then exits */
	MPI_Finalize();

	return 0;
}
