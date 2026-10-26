#!/usr/bin/env python3
"""
Module for a noiseless 1D Gaussian Process with RBF kernel,
prediction, and update methods.
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

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation
        of points in a Gaussian process.
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        # Mean formula: mu_s = K_s.T @ K_inv @ Y
        mu = np.dot(K_s.T, np.dot(K_inv, self.Y)).reshape(-1)

        # Variance formula: cov_s = K_ss - K_s.T @ K_inv @ K_s
        sigma = np.diagonal(K_ss - np.dot(K_s.T, np.dot(K_inv, K_s)))

        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Updates a Gaussian Process with a new sample point and value.
        """
        self.X = np.vstack((self.X, X_new.reshape(1, 1)))
        self.Y = np.vstack((self.Y, Y_new.reshape(1, 1)))
        self.K = self.kernel(self.X, self.X)
