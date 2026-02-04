#!/usr/bin/env python3
"""
This module contains a function to concatenate two 2D matrices along
a given axis.

The cat_matrices2D function returns a new matrix where mat1 and mat2
are combined along the specified axis (0 for rows, 1 for columns).
Returns None if concatenation is not possible.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two 2D matrices along a specific axis"""
    if axis == 0:
        # Check number of columns match
        if len(mat1) > 0 and len(mat2) > 0:
            if len(mat1[0]) != len(mat2[0]):
                return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    elif axis == 1:
        # Check number of rows match
        if len(mat1) != len(mat2):
            return None
        return [mat1[i][:] + mat2[i][:] for i in range(len(mat1))]
    else:
        return None
