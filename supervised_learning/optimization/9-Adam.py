#!/usr/bin/env python3
"""
Module to update variables using Adam optimization algorithm
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm

    Args:
        alpha: the learning rate
        beta1: the weight used for the first moment
        beta2: the weight used for the second moment
        epsilon: small number to avoid division by zero
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        v: the previous first moment of var
        s: the previous second moment of var
        t: the time step used for bias correction

    Returns:
        The updated variable, the new first moment, 
        and the new second moment, respectively
    """
    # 1. Update first moment (Momentum)
    new_v = (beta1 * v) + ((1 - beta1) * grad)

    # 2. Update second moment (RMSProp)
    new_s = (beta2 * s) + ((1 - beta2) * (grad ** 2))

    # 3. Bias correction for first moment
    v_corrected = new_v / (1 - (beta1 ** t))

    # 4. Bias correction for second moment
    s_corrected = new_s / (1 - (beta2 ** t))

    # 5. Update the variable
    new_var = var - (alpha * (v_corrected / (np.sqrt(s_corrected) + epsilon)))

    return new_var, new_v, new_s
