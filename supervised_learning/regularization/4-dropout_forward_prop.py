#!/usr/bin/env python3
"""
Module to conduct forward propagation with Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

    Args:
        X: numpy.ndarray (nx, m) containing the input data
        weights: dictionary of the weights and biases
        L: number of layers in the network
        keep_prob: probability that a node will be kept

    Returns:
        A dictionary containing the outputs and dropout masks
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights["W{}".format(i)]
        b = weights["b{}".format(i)]
        A_prev = cache["A{}".format(i - 1)]

        # Linear activation step
        Z = np.matmul(W, A_prev) + b

        if i == L:
            # Softmax for the output layer
            exp_Z = np.exp(Z)
            cache["A{}".format(i)] = exp_Z / np.sum(exp_Z, axis=0,
                                                    keepdims=True)
        else:
            # Tanh for hidden layers
            A = np.tanh(Z)
            # Create dropout mask: 1 if < keep_prob, 0 otherwise
            mask = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            # Apply mask and scale (Inverted Dropout)
            A = (A * mask) / keep_prob

            cache["D{}".format(i)] = mask.astype(int)
            cache["A{}".format(i)] = A

    return cache
