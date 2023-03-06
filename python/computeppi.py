#!/usr/bin/env python
# File: computeppi.py
# Name: D.Saravanan
# Date: 06/03/2023

""" Script to compute pi with mpi4py """

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def computepi(N):
    """compute pi"""
    dx = 1 / N  # step size

    workloads = [N // size for _ in range(size)]
    for n in range(N % size):
        workloads[n] += 1

    sload = 0
    for n in range(rank):
        sload += workloads[n]
    eload = sload + workloads[rank]

    x = lambda i: (i + 0.5) * dx
    pi = sum(map(lambda i: 4.0 / (1 + x(i) ** 2) * dx, range(sload, eload + 1)))
    pi = comm.reduce(pi, op=MPI.SUM, root=0)

    if rank == 0:
        return pi


if __name__ == "__main__":
    ns = 100000000  # number of steps
    pi = computepi(ns)
    if rank == 0:
        print(pi)
