/* File: mpiprog03.c
 * Name: D.Saravanan
 * Date: 02/12/2023
 * Program with non-blocking send/receive routines
*/

#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {

    int rank, size;
    int length, mate, mesg;
    char name[MPI_MAX_PROCESSOR_NAME];

    MPI_Status status[2];
    MPI_Request request[2];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size % 2 != 0) {
		if (rank == 0)
			printf("Quit!. Need an even number of processor, but nproc = %d\n", size);
    }

    else {
		if (rank == 0)
			printf("Number of processor: %d\n", size);

		MPI_Get_processor_name(name, &length);
		printf("rank %d of nproc %d on %s\n", rank, size, name);

		if (rank < size/2)
			mate = size/2 + rank;
		else if (rank >= size/2)
			mate = rank - size/2;

		/* nonblocking send */
		MPI_Isend(&rank, 1, MPI_INT, mate, 1, MPI_COMM_WORLD, &request[0]);

		/* nonblocking recv */
		MPI_Irecv(&mesg, 1, MPI_INT, mate, 1, MPI_COMM_WORLD, &request[1]);

		/* wait for all given communications to complete */
		MPI_Waitall(2, request, status);

		printf("rank %d is partner with %d\n", rank, mesg);
    }

	MPI_Finalize();

    return 0;
}
