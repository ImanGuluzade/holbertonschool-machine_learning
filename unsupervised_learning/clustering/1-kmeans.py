#!/usr/bin/env python3
"""Module that performs K-means on a dataset."""
import numpy as np

initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset

    Arguments:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters
        iterations: positive integer containing maximum number of iterations

    Returns:
        C: numpy.ndarray of shape (k, d) containing centroid means
        clss: numpy.ndarray of shape (n,) containing cluster index per point
        or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    # Initialize centroids using 0-initialize (uses numpy.random.uniform once)
    C = initialize(X, k)
    if C is None:
        return None, None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)

    for _ in range(iterations):
        # Compute distances between all points and centroids: shape (n, k)
        # Using broadcasting: (X[:, np.newaxis, :] - C[np.newaxis, :, :])
        distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        C_prev = C.copy()

        # Update centroids
        for i in range(k):
            points_in_cluster = X[clss == i]
            if len(points_in_cluster) == 0:
                # Reinitialize empty cluster centroid using numpy.random.uniform
                # (second allowed usage)
                C[i] = np.random.uniform(low=low, high=high, size=(d,))
            else:
                C[i] = np.mean(points_in_cluster, axis=0)

        # Check for convergence (no change in centroids)
        if np.all(C == C_prev):
            break

    return C, clss
