/* File: mpiprog01.c
 * Name: D.Saravanan
 * Date: 08/11/2022
 * Program to print the rank of the current process, the number of MPI
 * processes and the hostname on which the current process is running
*/

#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {

    int rank, size, length;
    char name[MPI_MAX_PROCESSOR_NAME];

    /* Initializes after program start, including program arguments */
    MPI_Init(&argc, &argv);

    /* Gets the rank number of the process */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Gets the number of ranks in the program determined by the mpiexec command */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Gets the name of the processor */
    MPI_Get_processor_name(name, &length);

    printf("rank %d of nproc %d on %s\n", rank, size, name);

    /* Finalizes MPI to synchronize ranks and then exits */
    MPI_Finalize();

    return 0;
}
