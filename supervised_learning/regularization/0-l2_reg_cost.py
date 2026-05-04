#!/usr/bin/env python3
"""
Module to calculate the cost of a neural network with L2 regularization
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization

    Args:
        cost: cost of the network without L2 regularization
        lambtha: regularization parameter
        weights: dictionary of the weights and biases (numpy.ndarrays)
        L: number of layers in the neural network
        m: number of data points used

    Returns:
        The cost of the network accounting for L2 regularization
    """
    l2_sum = 0
    for i in range(1, L + 1):
        key = "W{}".format(i)
        l2_sum += np.linalg.norm(weights[key])**2

    l2_cost = cost + (lambtha / (2 * m)) * l2_sum

    return l2_cost
