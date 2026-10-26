#!/usr/bin/env python3
"""Module that calculates the Q affinities for t-SNE"""
import numpy as np


def Q_affinities(Y):
    """
    Calculates the Q affinities

    Arguments:
        Y: numpy.ndarray of shape (n, ndim) containing the low
            dimensional transformation of X
            n is the number of points
            ndim is the new dimensional representation of X

    Returns:
        Q, num
        Q: numpy.ndarray of shape (n, n) containing the Q affinities
        num: numpy.ndarray of shape (n, n) containing the numerator
            of the Q affinities
    """
    sum_Y = np.sum(np.square(Y), axis=1)
    D = sum_Y[:, np.newaxis] + sum_Y[np.newaxis, :]
    D = D - 2 * np.matmul(Y, Y.T)
    num = 1 / (1 + D)
    np.fill_diagonal(num, 0)
    Q = num / np.sum(num)
    return Q, num
