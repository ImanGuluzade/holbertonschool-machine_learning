#!/usr/bin/env python3
"""Module that performs PCA on a dataset"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset

    Arguments:
        X: numpy.ndarray of shape (n, d) where
            n is the number of data points
            d is the number of dimensions in each point
            all dimensions have a mean of 0 across all data points
        var: fraction of the variance that the PCA transformation
            should maintain

    Returns:
        W: numpy.ndarray of shape (d, nd), the weights matrix that
            maintains var fraction of X's original variance, where
            nd is the new dimensionality of the transformed X
    """
    _, S, Vt = np.linalg.svd(X)
    ratios = np.cumsum(S) / np.sum(S)
    nd = np.argwhere(ratios >= var)[0, 0] + 1
    W = Vt[:nd].T
    return W
