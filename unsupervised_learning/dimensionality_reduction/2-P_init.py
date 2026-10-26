#!/usr/bin/env python3
"""Module that initializes variables for the t-SNE P affinities"""
import numpy as np


def P_init(X, perplexity):
    """
    Initializes all variables required to calculate the P affinities
    in t-SNE

    Arguments:
        X: numpy.ndarray of shape (n, d) containing the dataset
            n is the number of data points
            d is the number of dimensions in each point
        perplexity: the perplexity that all Gaussian distributions
            should have

    Returns:
        (D, P, betas, H)
        D: numpy.ndarray of shape (n, n), the squared pairwise
            distances between data points (diagonal is 0)
        P: numpy.ndarray of shape (n, n) of zeros, for the P affinities
        betas: numpy.ndarray of shape (n, 1) of ones, for the betas
        H: the Shannon entropy for perplexity, with a base of 2
    """
    n, _ = X.shape
    sum_X = np.sum(np.square(X), axis=1)
    D = sum_X[:, np.newaxis] + sum_X[np.newaxis, :]
    D = D - 2 * np.matmul(X, X.T)
    np.fill_diagonal(D, 0)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)
    return D, P, betas, H
