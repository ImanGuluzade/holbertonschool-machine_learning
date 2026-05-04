#!/usr/bin/env python3
"""
Updates weights with Dropout regularization using gradient descent
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization

    Args:
        Y: one-hot numpy.ndarray (classes, m) containing correct labels
        weights: dictionary of the weights and biases
        cache: dictionary of the outputs and dropout masks
        alpha: learning rate
        keep_prob: probability that a node will be kept
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

        # Calculate gradients for weights and biases
        dw = np.matmul(dz, A_prev.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        if i > 1:
            # Backprop through tanh activation: g'(Z) = 1 - A^2
            dg = 1 - (A_prev ** 2)
            # Calculate dz for the previous layer
            dz = np.matmul(W.T, dz) * dg
            # Apply the same mask used in forward prop and scale
            dz = (dz * cache['D' + str(i - 1)]) / keep_prob

        # Update weights and biases in place
        weights[W_key] = weights[W_key] - (alpha * dw)
        weights[B_key] = weights[B_key] - (alpha * db)
