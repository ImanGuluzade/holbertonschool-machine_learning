#!/usr/bin/env python3
"""
Updates weights using gradient descent with L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient
    descent with L2 regularization

    Args:
        Y: one-hot numpy.ndarray (classes, m) with correct labels
        weights: dictionary of weights and biases
        cache: dictionary of the outputs of each layer
        alpha: learning rate
        lambtha: L2 regularization parameter
        L: number of layers of the network
    """
    m = Y.shape[1]
    # Initial error at the output layer (Softmax + Cross-Entropy)
    dz = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W_key = 'W' + str(i)
        B_key = 'b' + str(i)
        W = weights[W_key]

        # dw calculation with L2 regularization term
        dw = (np.matmul(dz, A_prev.T) / m) + (lambtha / m) * W
        db = np.sum(dz, axis=1, keepdims=True) / m

        # Calculate dz for the next (previous) layer if not at the input
        if i > 1:
            # Derivative of tanh: 1 - A^2
            dg = 1 - (A_prev ** 2)
            dz = np.matmul(W.T, dz) * dg

        # Update weights and biases in place
        weights[W_key] = weights[W_key] - (alpha * dw)
        weights[B_key] = weights[B_key] - (alpha * db)
