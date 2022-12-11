#!/usr/bin/env python
# File: mpi4py3.py
# Name: D.Saravanan
# Date: 21/10/2022

""" Send multiple data items from one process to another with data tagging """

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    shared1 = {"d1": 55, "d2": 42}
    shared2 = {"d3": 25, "d4": 22}
    comm.send(shared1, dest=1, tag=1)
    comm.send(shared2, dest=1, tag=2)
    comm.send(shared1, dest=2, tag=1)
    comm.send(shared2, dest=2, tag=2)
    print(f"From rank {rank}, we sent {shared1} and {shared2}")

elif rank in (1, 2):
    receive1 = comm.recv(source=0, tag=1)
    receive2 = comm.recv(source=0, tag=2)
    print(f"On rank {rank}, we received {receive1} and {receive2}")

else:
    receive = 0.0
    print(f"On rank {rank}, receive is {receive}")
