#!/usr/bin/env python3
"""
This module contains a function to perform matrix multiplication
on numpy.ndarrays.

The np_matmul function returns a new numpy.ndarray that is the
matrix product of mat1 and mat2.
"""

import numpy as np


def np_matmul(mat1, mat2):
    """Performs matrix multiplication on two numpy.ndarrays"""
    return np.matmul(mat1, mat2)
