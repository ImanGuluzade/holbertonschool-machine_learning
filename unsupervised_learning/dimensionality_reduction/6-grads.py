#!/usr/bin/env python3
"""Module that calculates the gradients of Y for t-SNE"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates the gradients of Y

    Arguments:
        Y: numpy.ndarray of shape (n, ndim) containing the low
            dimensional transformation of X
        P: numpy.ndarray of shape (n, n) containing the P affinities
            of X

    Returns:
        (dY, Q)
        dY: numpy.ndarray of shape (n, ndim) containing the
            gradients of Y
        Q: numpy.ndarray of shape (n, n) containing the Q affinities
            of Y
    """
    Q, num = Q_affinities(Y)
    PQ = (P - Q) * num
    dY = np.sum(PQ, axis=1)[:, np.newaxis] * Y - np.matmul(PQ, Y)
    return dY, Q
