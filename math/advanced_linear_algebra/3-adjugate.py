#!/usr/bin/env python3
"""Calculates the adjugate matrix of a square matrix"""


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
            [matrix[i][j] for j in range(n) if j != c]
            for i in range(1, n)
        ]
        det += ((-1) ** c) * matrix[0][c] * determinant(minor_matrix)

    return det


def minor(matrix):
    """Calculates the minor matrix of a square matrix"""
    if not isinstance(matrix, list) or any(
            not isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    minor_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            submatrix = [
                [matrix[r][c] for c in range(n) if c != j]
                for r in range(n)
                if r != i
            ]
            row.append(determinant(submatrix))
        minor_matrix.append(row)

    return minor_matrix


def cofactor(matrix):
    """Calculates the cofactor matrix of a square matrix"""
    m = minor(matrix)
    n = len(m)
    cofactor_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(m[i][j] * (-1) ** (i + j))
        cofactor_matrix.append(row)
    return cofactor_matrix


def adjugate(matrix):
    """Calculates the adjugate matrix (transpose of cofactor matrix)"""
    c = cofactor(matrix)
    n = len(c)
    adj = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(c[j][i])
        adj.append(row)
    return adj
