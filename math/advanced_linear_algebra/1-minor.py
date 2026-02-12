#!/usr/bin/env python3
"""Calculates the minor matrix of a square matrix"""


def determinant(matrix):
    """Helper function to calculate determinant of a square matrix"""
    if matrix == [[]]:
        return 1
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for c in range(n):
        minor_matrix = [
            [matrix[i][j] for j in range(n) if j != c] for i in range(1, n)
        ]
        det += ((-1) ** c) * matrix[0][c] * determinant(minor_matrix)

    return det


def minor(matrix):
    """Calculates the minor matrix of a square matrix"""
    # Check if matrix is a list of lists
    if not isinstance(matrix, list) or any(not isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if non-empty square matrix
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Base case: 1x1 matrix
    if n == 1:
        return [[1]]

    # Build the minor matrix
    minor_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            # Create submatrix removing row i and column j
            submatrix = [
                [matrix[r][c] for c in range(n) if c != j]
                for r in range(n)
                if r != i
            ]
            row.append(determinant(submatrix))
        minor_matrix.append(row)

    return minor_matrix
