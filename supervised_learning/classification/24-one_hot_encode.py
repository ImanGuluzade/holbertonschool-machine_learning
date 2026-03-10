#!/usr/bin/env python3
"""Module for one-hot encoding"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix
    Y: numpy.ndarray of shape (m,) containing numeric class labels
    classes: the maximum number of classes
    Returns: one-hot matrix of shape (classes, m) or None on failure
    """
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None
    if not isinstance(classes, int) or classes <= np.max(Y):
        return None

    try:
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        # Use advanced indexing to place 1s at the correct locations
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except Exception:
        return None
