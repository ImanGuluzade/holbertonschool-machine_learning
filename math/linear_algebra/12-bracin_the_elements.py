#!/usr/bin/env python3
"""
This module contains a function to perform element-wise operations
on numpy.ndarrays.

The np_elementwise function returns a tuple containing the sum,
difference, product, and quotient of two matrices.
"""


def np_elementwise(mat1, mat2):
    """Performs element-wise addition, subtraction, multiplication,
    and division on two numpy.ndarrays and returns them as a tuple"""
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
