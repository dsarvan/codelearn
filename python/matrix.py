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


# Matrix Transpose:
# The matrix transpose flips a matrix over its diagonal. It swaps rows with columns.

transpose_A = np.transpose(A)
transpose_B = np.transpose(B)

print(f"Transpose of the Matrix A:\n {A.T}\n")
print(f"Transpose of the Matrix B:\n {B.T}\n")


# Eigenvalues and Eigenvectors:
# Eigenvalues show the extent to which an eigenvector is scaled during transformation.
# Eigenvectors do not change direction under this transformation.

evals, evecs = np.linalg.eig(A)

print(f"Eignevalues:\n {evals}\n")
print(f"Eigenvectors:\n {evecs}\n")


# LU Decomposition:
# LU decomposition breaks a matrix into two parts. One part is a lower triangular
# matrix (L). The other part is an upper triangular matrix (U). It helps solve
# linear least squares problems and find eigenvalues.

from scipy.linalg import lu

P, L, U = lu(A) # A = P @ L @ U

print("LU Decomposition: A = P @ L @ U")
print(f"Permutation Matrix P:\n {P}\n")
print(f"Lower triangular Matrix (unit diagonal elements) L:\n {L}\n")
print(f"Upper triangular Matrix U:\n {U}\n")


# QR Decomposition:
# QR decomposition divides a matrix into two parts. One part is an orthogonal
# matrix (Q). The other part is an upper triangular matrix (R). It helps solve
# linear least squares problems and find eigenvalues.

from scipy.linalg import qr

Q, R = qr(A) # A = Q R

print("QR Decomposition: A = Q R")
print(f"Unitary/Orthogonal Matrix Q:\n {Q}\n")
print(f"Upper triangular Matrix R:\n {R}\n")


# SVD (Singular Value Decomposition):
# SVD decomposes a matrix into three matrices: U, S, and Vh. U and Vh are orthogonal
# matrices. S is a diagonal matrix. It is useful in many applications like data
# reduction and solving linear systems.

from scipy.linalg import svd

U, S, Vh = svd(A) # A = U @ S @ Vh, S is matrix of zeros with main diagonal s

print("SVD Decomposition: A = U @ S @ Vh")
print(f"Unitary Matrix U:\n {U}\n")
print(f"Singular values (real, non-negative) s:\n {S}\n")
print(f"Unitary Matrix Vh:\n {Vh}\n")


# Solution of Linear Equations:
# Find the values of variables that satisfy equations in a system. Each equation
# represents a straight line. The solution is where these lines meet.

# 4x + 2y + 4z = 44
# 5x + 3y + 7z = 56
# 9x + 3y + 6z = 72

# using NumPy to solve the system of linear equations:

A = np.array([[4, 2, 4], [5, 3, 7], [9, 3, 6]]) # matrix A
B = np.array([44, 56, 72]) # vector B

# solve the system of linear equations AX = B
X = np.linalg.solve(A, B) # X = np.linalg.inv(A).dot(B)

print(f"Solution to the system of linear equations AX = B: {X}\n")

# using SymPy to solve the system of linear equations:

from sympy import Matrix

A = Matrix([[4, 2, 4], [5, 3, 7], [9, 3, 6]]) # matrix A
B = Matrix([44, 56, 72]) # vector B

# solve the system of linear equations AX = B
X = A.inv() * B

print(f"Solution to the system of linear equations AX = B: {X}\n")


# Least-Squares Fitting:
# The least squares fitting finds the best match for data points. It lowers the
# squared differences between actual and predicted values.

A = np.array([[1, 1], [1, 2], [1, 3]]) # matrix A
B = np.array([1, 2, 2]) # vector B

# solve the linear matrix equation least-squares problem A @ x = B
x, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)

print(f"Least Squares solution: {x}")
print(f"Residuals: {residuals}")
print(f"Rank of the matrix: {rank}")
print(f"Singular values: {s}\n")


# Matrix Norms:
# Matrix norms measure the size of a matrix. Norms are useful to check numerical
# stability and analyze matrices.

A = np.array([[4, 2, 4], [5, 3, 7], [9, 3, 6]]) # matrix A

# compute various norms
onenrm = np.linalg.norm(A, 1)
froben = np.linalg.norm(A, 'fro')
infint = np.linalg.norm(A, np.inf)

print(f"1-Norm: {onenrm}")
print(f"Frobenius: {froben}")
print(f"InfinityN: {infint}\n")


# Condition Number:
# The condition number of a matrix measures sensitivity to input changes. A high
# condition number means the solution could be unstable.

A = np.array([[4, 2, 4], [5, 3, 7], [9, 3, 6]]) # matrix A

# compute the condition number of the matrix
condition_number = np.linalg.cond(A)

print(f"Condition Number: {condition_number}\n")
