#!/usr/bin/env python3
"""
This module contains a function to calculate the shape of a matrix.

The matrix_shape function returns a list of integers representing
the dimensions of the matrix.
"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix as a list of integers"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return shape
