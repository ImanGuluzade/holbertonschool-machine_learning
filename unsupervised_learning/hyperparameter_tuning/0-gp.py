#!/usr/bin/env python3
"""
Module for a noiseless 1D Gaussian Process with RBF kernel.
"""

import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian process.
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initializes the Gaussian Process.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices using RBF.
        """
        term1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        term2 = np.sum(X2 ** 2, axis=1)
        term3 = 2 * np.dot(X1, X2.T)
        sqdist = term1 + term2 - term3
        return (self.sigma_f ** 2) * np.exp(-0.5 / (self.l ** 2) * sqdist)
