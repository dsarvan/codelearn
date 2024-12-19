#!/usr/bin/env python
# File: devicedata.py
# Name: D.Saravanan
# Date: 19/12/2024
# Script access information on device metadata and occupancy

import pycuda.autoinit
from pycuda.tools import DeviceData
from pycuda.tools import OccupancyRecord


# device metadata
# pycuda.tools.DeviceData(dev=None)
# Gives access to more information on a device than is available through
# pycuda.driver.Device.get_attribute(). If dev is None, it defaults to the
# device returned by pycuda.driver.Context.get_device()

# maximum threads
max_threads = DeviceData().max_threads

# warp size
warp_size = DeviceData().warp_size

# warps per multiprocessor
warps_per_mp = DeviceData().warps_per_mp

# thread blocks per multiprocessor
thread_blocks_per_mp = DeviceData().thread_blocks_per_mp

# number of registers
registers = DeviceData().registers

# shared memory
shared_memory = DeviceData().shared_memory

# number of threads that participate in banked, simultaneous access to shared memory
smem_granularity = DeviceData().smem_granularity

# size of the smallest possible (non-empty) shared memory allocation
smem_alloc_granularity = DeviceData().smem_alloc_granularity

# distance between global memory base addresses that allow accesses of word-size
# word_size bytes to get coalesced
align_bytes = DeviceData().align_bytes()

# round up bytes to the next alignment boundary as given by align_bytes()
align =  DeviceData().align(bytes=DeviceData().align_bytes())


# occupancy
# pycuda.tools.OccupancyRecord(devdata, threads, shared_mem=0, registers=0)
# Calculate occupancy for a given kernel workload characterized by
# * thread count of threads
# * shared memory use of shared_mem bytes
# * register use of registers 32-bit registers

# how many thread blocks execute on each multiprocessor?
tb_per_mp = OccupancyRecord(DeviceData(), DeviceData().max_threads).tb_per_mp

# what tb_per_mp is limited by? One of "device", "warps", "regs", "smem".
limited_by = OccupancyRecord(DeviceData(), DeviceData().max_threads).limited_by

# how many warps execute on each multiprocessor?
warps_per_mp = OccupancyRecord(DeviceData(), DeviceData().max_threads).warps_per_mp

# float value between 0 and 1 indicating how much of each multiprocessor's
# scheduling capability is occupied by the kernel.
occupancy = OccupancyRecord(DeviceData(), DeviceData().max_threads).occupancy
