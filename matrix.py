#!/usr/bin/env python
# File: matrix.py
# Name: D.Saravanan
# Date: 01/09/2021

""" Script for matrix manipulation """

import numpy as np

# matrix assignment
A = np.array([[1, 3 - 1j], [3j, -1 + 1j]])
print(A)

# matrix transpose
print(A.T)

# Hermitian conjugate
print(A.conj().T)

# matrix inverse
print(np.linalg.inv(A))

# trace
print(np.trace(A))

# determinant
print(np.linalg.det(A))

# eigenvalues, eigenvectors
vals, vecs = np.linalg.eig(A)
print(vals, vecs, sep="\n\n")
