#!/usr/bin/env python3
"""Module that performs PCA on a dataset"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset

    Arguments:
        X: numpy.ndarray of shape (n, d) where
            n is the number of data points
            d is the number of dimensions in each point
        ndim: the new dimensionality of the transformed X

    Returns:
        T: numpy.ndarray of shape (n, ndim) containing the
            transformed version of X
    """
    X_m = X - np.mean(X, axis=0)
    _, _, Vt = np.linalg.svd(X_m, full_matrices=False)
    W = Vt[:ndim].T
    T = np.matmul(X_m, W)
    return T
