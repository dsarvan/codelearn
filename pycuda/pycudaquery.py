#!/usr/bin/env python
# File: pycudaquery.py
# Name: D.Saravanan
# Date: 21/12/2024
# Script query CUDA device with PyCUDA

import pycuda.autoinit
import pycuda.driver as drv

print("CUDA device query (PyCUDA version)")

ndevice = drv.Device.count()
print(f"Detected {ndevice} CUDA capable device(s)")

for n in range(ndevice):
	print(f"\nDEVICE {n}: {drv.Device(n).name()}")
	print(f"COMPUTE CAPABILITY: {device.compute_capability()}")
	print(f"TOTAL MEMORY: {device.total_memory()//(1024**2)} megabytes")

	for key, value in device.get_attributes().items():
		print(f"{key}: {value}")
