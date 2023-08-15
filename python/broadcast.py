#!/usr/bin/env python
# File: broadcast.py
# Name: D.Saravanan
# Date: 07/03/2023

""" Script to broadcast data to all processes """

from mpi4py import MPI
from numpy import array


def main():
    """broadcast data to all processes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        data = {
            "key1": [10, 10.1, 10 + 11j],
            "key2": ["mpi4py", "python"],
            "key3": array([1, 2, 3]),
        }
    else:
        data = None

    # broadcast data to all processes
    data = comm.bcast(data, root=0)

    if rank == 0:
        print("bcast finished")

    print(f"data on rank {rank} of size {size} is {data}")
    comm.Barrier()


if __name__ == "__main__":
    main()
