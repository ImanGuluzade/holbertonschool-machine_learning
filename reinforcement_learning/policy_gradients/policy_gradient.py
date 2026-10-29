#!/usr/bin/env python3
"""Module that computes the policy with a weight of a matrix."""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix

    Arguments:
        matrix: numpy.ndarray representing the states
        weight: numpy.ndarray representing the weights

    Returns:
        The policy (action probabilities) as a numpy.ndarray
    """
    z = np.dot(matrix, weight)
    # Exponentiate with numerical stability
    e_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e_z / np.sum(e_z, axis=-1, keepdims=True)
