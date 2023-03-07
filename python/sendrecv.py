#!/usr/bin/env python
# File: sendrecv.py
# Name: D.Saravanan
# Date: 08/03/2023

""" Script to create an array in rank 0 process and 
send the first part of the array to the rank 1 process 
and the second part of the array to the rank 2 process """

import numpy as np
from mpi4py import MPI


def main():
    """point-to-point communication"""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        data = np.arange(10)
        comm.send(data[:5], dest=1, tag=11)
        comm.send(data[5:], dest=2, tag=12)
        print(f"Rank {rank} data is {data}")
    elif rank == 1:
        data = comm.recv(source=0, tag=11)
        print(f"Rank {rank} received data {data}")
    elif rank == 2:
        data = comm.recv(source=0, tag=12)
        print(f"Rank {rank} received data {data}")
    else:
        data = None


if __name__ == "__main__":
    main()
