#!/usr/bin/env python
# File: devicequery.py
# Name: D.Saravanan
# Date: 11/03/2023

""" Script to find number of gpu devices on host computer """

import pycuda.driver as drv
import pycuda.autoinit

# number of gpu devices on host computer
print(f"Detected {drv.Device.count()} CUDA capable devices.")

for n in range(drv.Device.count()):
    device = drv.Device(n)
    print(f"Device {n}: {device.name()}")

    compute = float("%d.%d" % device.compute_capability())
    print(f"Compute capability: {compute}")

    print(f"Total Memory: {device.total_memory()//(1024**2)} megabytes")

    # attributes of gpu device
    device_attributes_tuples = device.get_attributes().iteritems()
    device_attributes = {}

    for k, v in device_attributes_tuples:
        device_attributes[str(k)] = v

    sm = device_attributes["MULTIPROCESSOR_COUNT"]
    print(sm)
