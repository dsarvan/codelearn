#!/usr/bin/env python
# File: mpiprog02.py
# Name: D.Saravanan
# Date: 01/12/2023

""" Script with blocking send/receive routines """

from mpi4py import MPI


def main():
    """blocking communication"""

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
            comm.send(rank, dest=mate, tag=11)
            mesg = comm.recv(source=mate, tag=12, status=MPI.Status())

        elif rank >= size / 2:
            mate = rank - size / 2
            mesg = comm.recv(source=mate, tag=11, status=MPI.Status())
            comm.send(rank, dest=mate, tag=12)

        print(f"rank {rank} is partner with {mesg}")


if __name__ == "__main__":
    main()
