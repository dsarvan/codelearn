#!/usr/bin/env python
# File: devicename.py
# Name: D.Saravanan
# Date: 17/12/2024
# Script query CUDA capable devices and print device name

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.tools import make_default_context

print("/* number of devices */")
ndevice = drv.Device.count()
print(f"Detected {ndevice} CUDA capable device(s)")

print("\n/* device name with pycuda driver */")
for n in range(ndevice):
	device = drv.Device(n)
	print(f"Device {n}: {device.name()}")

print("\n/* device name with pycuda tools */")
device = make_default_context().get_device()
print(f"Device name: {device.name()}")
