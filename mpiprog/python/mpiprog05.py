#!/usr/bin/env python
# File: mpiprog05.py
# Name: D.Saravanan
# Date: 04/12/2023

""" Script to initialize matrices, assign values, demonstrates
data decomposition and compute matrix multiplication """

import numpy as np
from mpi4py import MPI


def main():
    """matrix multiplication"""

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    nrow, nval, ncol = 10062, 10015, 1007

    if nrow % size != 0:
        if rank == 0:
            print(f"Matrix size not divisible by process count {size}.")
        MPI.Finalize()
        return 0

    if rank == 0:
        print("Starting parallel matrix multiplication ...")
        print(f"Using matrix sizes A[{nrow}][{nval}], B[{nval}][{ncol}], C[{nrow}][{ncol}]")
        print("Initializing matrices ...")

    A = (np.array([[i + j for j in range(nval)] for i in range(nrow)], dtype=float) if rank == 0 else None) # matrix A
    B = (np.array([[i * j for j in range(nval)] for i in range(ncol)], dtype=float) if rank == 0 else None) # matrix B

    workload_A = [nrow // size for _ in range(size)]
    for n in range(nrow % size): workload_A[n] += 1

    workload_B = [ncol // size for _ in range(size)]
    for n in range(ncol % size): workload_B[n] += 1

    mrow = workload_A[rank] # no. of rows in matrix nA
    mcol = workload_B[rank] # no. of cols in matrix nB

    count_A = [workload_A[i] * nval for i in range(size)] # no. of elements in nA
    displ_A = [sum(count_A[:n]) for n in range(size)] # displacement data nA

    nA = np.zeros((mrow, nval)) # matrix nA

	# scatter matrix A elements from process 0 to all other process in communicator
    comm.Scatterv([A, count_A, displ_A, MPI.DOUBLE], nA, root=0)

    count_B = [nval * workload_B[i] for i in range(size)] # no. of elements in nB
    displ_B = [sum(count_B[:n]) for n in range(size)] # displacement data nB

    nB = np.zeros((mcol, nval)) # matrix nB

	# scatter matrix B elements from process 0 to all other process in communicator
    comm.Scatterv([B, count_B, displ_B, MPI.DOUBLE], nB, root=0)

    nC = np.zeros((mcol, nrow)) # matrix nC

    rval = rank

    # matrix multiplication
    if rank == 0: print("Performing matrix multiplication ...")
    for iter in range(size):

        sval = nA @ nB.transpose()
        nC[0:mcol, rval * mrow : rval * mrow + mrow] = sval.transpose()
        rval = 0 if rval == size - 1 else rval + 1

        # blocking send and receive routines
        comm.Sendrecv_replace(nA,
            size - 1 if rank == 0 else rank - 1, 11,
            0 if rank == size - 1 else rank + 1, 11,
            status=MPI.Status(),
        )

    count_C = [nrow * workload_B[i] for i in range(size)] # no. of elements in nC
    displ_C = [sum(count_C[:n]) for n in range(size)] # displacement data nC

    M = np.zeros((ncol, nrow)) if rank == 0 else None # matrix M

    # gather matrix C elements to process 0 from all other process in communicator
    comm.Gatherv(nC, [M, count_C, displ_C, MPI.DOUBLE], root=0)

    C = np.zeros((ncol, nrow)) if rank == 0 else None # matrix C

    # print matrix result
    if rank == 0:
        C = M.transpose()
        print("Here is the result matrix:")
        print(C)

    MPI.Finalize()


if __name__ == "__main__":
    main()
