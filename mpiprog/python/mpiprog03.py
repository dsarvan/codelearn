#!/usr/bin/env python
# File: mpiprog03.py
# Name: D.Saravanan
# Date: 02/12/2023

""" Script with non-blocking send/receive routines """

from mpi4py import MPI


def main():
    """non-blocking communication"""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size % 2 != 0:
        if rank == 0:
            print(f"Quit! Need an even number of processes, but nproc is {size}.")

    else:
        if rank == 0:
            print(f"Number of processes: {size}")

        name = MPI.Get_processor_name()
        print(f"rank {rank} of nproc {size} on {name}")

        if rank < size / 2:
            mate = size / 2 + rank
        elif rank >= size / 2:
            mate = rank - size / 2

        # nonblocking send
        request = comm.isend(rank, dest=mate, tag=11)
        request.wait()

        # nonblocking recv
        request = comm.irecv(source=mate, tag=11)
        mesg = request.wait()

        print(f"rank {rank} is partner with {mesg}")


if __name__ == "__main__":
    main()
