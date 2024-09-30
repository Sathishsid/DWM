import numpy as np
# Create two arrays for operations
A = np.array([[3, 1], [1, 2]])
B = np.array([9, 8])

# a. Dot and matrix product of two arrays
dot_product = np.dot(A, B)
matrix_product = np.matmul(A, B)
print("Dot Product:")
print(dot_product)
print("Matrix Product:")
print(matrix_product)

# Create a matrix for subsequent operations
C = np.array([[3, 1], [1, 2]])

# b. Compute the Eigen values of a matrix
eigenvalues = np.linalg.eigvals(C)
print("\nEigenvalues:")
print(eigenvalues)

# c. Solve a linear matrix equation
coefficients = np.array([[3, 1], [1, 2]])
constants = np.array([9, 8])
solution = np.linalg.solve(coefficients, constants)
print("\nSolution to the linear matrix equation:")
print(solution)

# d. Compute the multiplicative inverse of a matrix
inverse = np.linalg.inv(C)
print("\nMultiplicative Inverse:")
print(inverse)

# e. Compute the rank of a matrix
rank = np.linalg.matrix_rank(C)
print("\nRank of the matrix:")
print(rank)

# f. Compute the determinant of an array
determinant = np.linalg.det(C)
print("\nDeterminant of the matrix:")
print(determinant)