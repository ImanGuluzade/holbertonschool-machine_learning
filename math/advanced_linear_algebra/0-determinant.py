#!/usr/bin/env python3
"""Calculates the determinant of a matrix"""


def determinant(matrix):
    """Calculates the determinant of a square matrix"""
    # Check if matrix is a list of lists
    if (not isinstance(matrix, list)
            or any(not isinstance(row, list) for row in matrix)):
        raise TypeError("matrix must be a list of lists")

    # 0x0 matrix
    if matrix == [[]]:
        return 1

    n = len(matrix)

    # Check if square
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base cases
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive calculation for n > 2
    det = 0
    for c in range(n):
        minor = [[matrix[i][j] for j in range(n) if j != c]
                 for i in range(1, n)]
        det += ((-1) ** c) * matrix[0][c] * determinant(minor)

    return det
