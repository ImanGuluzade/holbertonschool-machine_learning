#!/usr/bin/env python3
"""
Module to update variables using RMSProp optimization algorithm
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm

    Args:
        alpha: the learning rate
        beta2: the RMSProp weight (discounting factor)
        epsilon: small number to avoid division by zero
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        s: the previous second moment of var

    Returns:
        The updated variable and the new moment, respectively
    """
    # 1. Update the second moment (weighted moving average of squared grads)
    # s_new = beta2 * s_prev + (1 - beta2) * (gradient^2)
    new_s = (beta2 * s) + ((1 - beta2) * (grad ** 2))

    # 2. Update the variable
    # var_new = var_old - alpha * (grad / (sqrt(new_s) + epsilon))
    new_var = var - (alpha * (grad / (np.sqrt(new_s) + epsilon)))

    return new_var, new_s
