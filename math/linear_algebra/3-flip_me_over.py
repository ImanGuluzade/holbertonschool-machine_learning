#!/usr/bin/env python3
"""
This module contains a function to calculate the transpose of a 2D matrix.

The matrix_transpose function returns a new matrix where rows and columns
are swapped.
"""


def matrix_transpose(matrix):
    """Returns the transpose of a 2D matrix"""
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
