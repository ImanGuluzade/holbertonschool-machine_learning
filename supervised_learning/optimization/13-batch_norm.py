#!/usr/bin/env python3
"""
Module to perform batch normalization on a matrix
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network
    using batch normalization

    Args:
        Z: numpy.ndarray of shape (m, n) to be normalized
        gamma: numpy.ndarray of shape (1, n) containing the scales
        beta: numpy.ndarray of shape (1, n) containing the offsets
        epsilon: small number used to avoid division by zero

    Returns:
        The normalized Z matrix
    """
    # 1. Calculate the mean and variance along the batch (m) axis
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)

    # 2. Standardize Z (center and scale)
    # Z_norm = (Z - mu) / sqrt(var + epsilon)
    Z_centered = Z - mean
    Z_standardized = Z_centered / np.sqrt(variance + epsilon)

    # 3. Apply the learnable parameters gamma and beta
    # Z_final = gamma * Z_standardized + beta
    normalized_Z = gamma * Z_standardized + beta

    return normalized_Z
