#!/usr/bin/env python3
"""
Module to create mini-batches for training a neural network
"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches for training a neural network using
    mini-batch gradient descent

    Args:
        X: numpy.ndarray of shape (m, nx) representing input data
        Y: numpy.ndarray of shape (m, ny) representing the labels
        batch_size: the number of data points in a batch

    Returns:
        List of mini-batches containing tuples (X_batch, Y_batch)
    """
    # 1. Shuffle the data before batching
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    m = X.shape[0]
    mini_batches = []

    # 2. Iterate through data and slice into batches
    for i in range(0, m, batch_size):
        # Calculate the end index for the current batch
        end = i + batch_size
        if end > m:
            end = m

        X_batch = X_shuffled[i:end]
        Y_batch = Y_shuffled[i:end]

        mini_batches.append((X_batch, Y_batch))

    return mini_batches
