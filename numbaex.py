#!/usr/bin/env python3

from numba import jit
import numpy as np
import time

x = np.arange(100).reshape(10, 10)
y = np.arange(100).reshape(10, 10)

@jit(nopython=True)
def go_fast(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

start = time.time()
print(go_fast(x))
end = time.time()

print(end - start)

def ggo_fast(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

start = time.time()
print(ggo_fast(y))
end = time.time()

print(end - start)
