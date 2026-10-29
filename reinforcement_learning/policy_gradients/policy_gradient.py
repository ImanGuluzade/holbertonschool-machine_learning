#!/usr/bin/env python3
"""Module that computes the policy with a weight of a matrix."""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix

    Arguments:
        matrix: numpy.ndarray representing the state (or batch of states)
        weight: numpy.ndarray representing the policy weights

    Returns:
        The policy (action probabilities) as a numpy.ndarray
    """
    # Calculate the logits via dot product
    z = np.dot(matrix, weight)

    # Apply softmax with numerical stability (subtract max)
    e_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e_z / np.sum(e_z, axis=-1, keepdims=True)
