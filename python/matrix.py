#!/usr/bin/env python
# File: matrix.py
# Name: D.Saravanan
# Date: 01/09/2021

""" Script matrix manipulation for linear algebra applications """

import numpy as np

# Matrix Assignment:
A = np.array([[1, 3, 1], [3, -1, 10], [4, -2, -1]])
B = np.array([[4, 7, 3], [-2, 11, 5], [-3, 1, -6]])

print(f"Matrix A:\n {A}\n")
print(f"Matrix B:\n {B}\n")


# Matrix Multiplication:
# Matrix multiplication creates a new matrix from two matrices. Multiply each row of
# the first matrix with each column of the second matrix. Add the products to get each
# element in the new matrix.

C = np.dot(A, B)

print(f"Matrix multiplication A*B:\n {C}\n")


# Matrix Inversion:
# Multiply a matrix with its inverse to get the identity matrix. It helps solve systems
# of linear equations. Only square and non-singular matrices have an inverse.

A_inv = np.linalg.inv(A)
B_inv = np.linalg.inv(B)

print(f"Inverse of Matrix A:\n {A_inv}\n")
print(f"Inverse of Matrix B:\n {B_inv}\n")


# Matrix Determinant:
# Matrix determinant is a number from a matrix. It tells us if the matrix can be
# inverted. We use specific rules to calculate it based on matrix size.

det_A = np.linalg.det(A)
det_B = np.linalg.det(B)

print(f"Determinant of the Matrix A: {det_A}\n")
print(f"Determinant of the Matrix B: {det_B}\n")


# Matrix Trace:
# The trace is the sum of the diagonal elements. It only applies to square matrices.
# We get a single number as the trace.

trace_A = np.trace(A)
trace_B = np.trace(B)

print(f"Trace of the Matrix A: {trace_A}\n")
print(f"Trace of the Matrix B: {trace_B}\n")
