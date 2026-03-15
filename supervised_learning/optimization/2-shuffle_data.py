#!/usr/bin/env python3
"""
Module to shuffle data points in two matrices
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way

    Args:
        X: numpy.ndarray of shape (m, nx) to shuffle
           m is the number of data points
           nx is the number of features in X
        Y: numpy.ndarray of shape (m, ny) to shuffle
           m is the same number of data points as in X
           ny is the number of features in Y

    Returns:
        The shuffled X and Y matrices
    """
    # Generate a random permutation of indices from 0 to m-1
    m = X.shape[0]
    permutation = np.random.permutation(m)

    # Use the same permutation to reorder both X and Y
    return X[permutation], Y[permutation]
