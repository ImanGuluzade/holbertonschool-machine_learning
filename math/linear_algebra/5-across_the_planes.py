#!/usr/bin/env python3
"""
This module contains a function to add two 2D matrices element-wise.

The add_matrices2D function returns a new matrix where each element
is the sum of the corresponding elements in mat1 and mat2.
If the matrices are not the same shape, it returns None.
"""


def add_matrices2D(mat1, mat2):
    """Adds two 2D matrices element-wise"""
    if len(mat1) != len(mat2):
        return None
    if any(len(row1) != len(row2) for row1, row2 in zip(mat1, mat2)):
        return None
    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
            for i in range(len(mat1))]
