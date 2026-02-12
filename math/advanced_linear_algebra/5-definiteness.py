#!/usr/bin/env python3
"""Calculates the definiteness of a symmetric matrix using NumPy"""

import numpy as np


def definiteness(matrix):
    """Determines if a symmetric matrix is positive/negative definite or semi-definite"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # Check symmetry
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        eigvals = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None

    pos = np.all(eigvals > 0)
    pos_semi = np.all(eigvals >= 0) and np.any(eigvals == 0)
    neg = np.all(eigvals < 0)
    neg_semi = np.all(eigvals <= 0) and np.any(eigvals == 0)

    if pos:
        return "Positive definite"
    if pos_semi:
        return "Positive semi-definite"
    if neg:
        return "Negative definite"
    if neg_semi:
        return "Negative semi-definite"
    if np.any(eigvals > 0) and np.any(eigvals < 0):
        return "Indefinite"

    return None
