#!/usr/bin/env python3
"""
Module to normalize a dataset
"""
import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix

    Args:
        X: numpy.ndarray of shape (d, nx) to normalize
           d is the number of data points
           nx is the number of features
        m: numpy.ndarray of shape (nx,) containing the mean
           of all features of X
        s: numpy.ndarray of shape (nx,) containing the standard
           deviation of all features of X

    Returns:
        The normalized X matrix
    """
    # Formula: z = (x - mu) / sigma
    return (X - m) / s
