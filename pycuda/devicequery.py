#!/usr/bin/env python
# File: devicequery.py
# Name: D.Saravanan
# Date: 18/12/2024
# Script query device compute capability

import pycuda.autoinit
import pycuda.driver as drv

ndevice = drv.Device.count()
print(f"Detected {ndevice} CUDA capable device(s)")

free, total = drv.mem_get_info()
print(f"Memory occupancy: {free*100/total}% free")

for n in range(ndevice):
	device = drv.Device(n)
	attris = device.get_attributes()

	print(f"\n/* Attributes for device {n}: {device.name()} */")
	for key, value in attris.items():
		print(f"{key}: {value}")
