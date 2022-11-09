! File: mpiprog.f90
! Name: D.Saravanan
! Date: 08/11/2022
! MPI program working example

program parallel
include "mpif.h"
integer ierr, rank, size

! Initializes after program start,
! including program arguments
call MPI_INIT(ierr)

! Gets the rank number of the process
call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)

! Gets the number of ranks in the program
! determined by the mpiexec command
call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierr)

print *, "Hello world from process ", rank, " of ", size

! Finalizes MPI to synchronize ranks and then exists
call MPI_FINALIZE(ierr)

end program
