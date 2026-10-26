#!/usr/bin/env python3
"""Module that calculates the symmetric P affinities of a dataset"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a dataset

    Arguments:
        X: numpy.ndarray of shape (n, d) containing the dataset
            n is the number of data points
            d is the number of dimensions in each point
        tol: maximum tolerance allowed (inclusive) for the difference
            in Shannon entropy from perplexity
        perplexity: the perplexity that all Gaussian distributions
            should have

    Returns:
        P: numpy.ndarray of shape (n, n) containing the symmetric
            P affinities
    """
    n, _ = X.shape
    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        low = None
        high = None
        idx = np.concatenate((np.arange(0, i), np.arange(i + 1, n)))
        Di = D[i, idx]
        Hi, Pi = HP(Di, betas[i])
        Hdiff = Hi - H

        while np.abs(Hdiff) > tol:
            if Hdiff > 0:
                low = betas[i, 0]
                if high is None:
                    betas[i] = betas[i] * 2
                else:
                    betas[i] = (betas[i] + high) / 2
            else:
                high = betas[i, 0]
                if low is None:
                    betas[i] = betas[i] / 2
                else:
                    betas[i] = (betas[i] + low) / 2
            Hi, Pi = HP(Di, betas[i])
            Hdiff = Hi - H

        P[i, idx] = Pi

    P = (P + P.T) / (2 * n)
    return P
