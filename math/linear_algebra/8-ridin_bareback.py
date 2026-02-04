#!/usr/bin/env python3
"""
This module contains a function to perform matrix multiplication.

The mat_mul function returns a new matrix which is the product of mat1
and mat2. Returns None if the matrices cannot be multiplied.
"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication of two 2D matrices"""
    # Check if multiplication is possible
    if len(mat1[0]) != len(mat2):
        return None
    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            val = 0
            for k in range(len(mat2)):
                val += mat1[i][k] * mat2[k][j]
            row.append(val)
        result.append(row)
    return result
