#!/usr/bin/env python
# File: version.py
# Name: D.Saravanan
# Date: 20/12/2024
# Script query CUDA and PyCUDA version

import pycuda.autoinit
import pycuda.driver as drv

# The version of CUDA against which PyCUDA was compiled.
# Returns a 3-tuple of integers as (major, minor, revision).
print(f"CUDA version: {drv.get_version()}")

# The version of the CUDA driver on top of which PyCUDA is running.
# Returns an integer version number.
print(f"CUDA driver version: {drv.get_driver_version()}")

# The numeric version of PyCUDA as a variable-length tuple of integers.
# Enables easy version checks such as VERSION >= (0,93).
print(f"PyCUDA version: {pycuda.VERSION}")

# A text string such as "rc4" or "beta" qualifying the status of the release.
print(f"PyCUDA version status: {pycuda.VERSION_STATUS}")

# The full release name (such as "0.93rc4") in string form.
print(f"PyCUDA version text: {pycuda.VERSION_TEXT}")
