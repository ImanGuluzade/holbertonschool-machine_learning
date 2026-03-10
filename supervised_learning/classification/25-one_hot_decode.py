#!/usr/bin/env python3
"""Module for one-hot decoding"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels
    one_hot: one-hot encoded numpy.ndarray with shape (classes, m)
    Returns: numpy.ndarray with shape (m, ) containing the numeric labels,
             or None on failure
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    try:
        # argmax returns the indices of the maximum values along an axis
        # axis=0 corresponds to the classes (rows)
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
