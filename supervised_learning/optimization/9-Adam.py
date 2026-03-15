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
        epsilon: a small number to avoid division by zero
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
    v_new = (beta1 * v) + ((1 - beta1) * grad)

    # 2. Update second moment (RMSProp)
    s_new = (beta2 * s) + ((1 - beta2) * (grad ** 2))

    # 3. Bias correction for first moment
    v_corrected = v_new / (1 - (beta1 ** t))

    # 4. Bias correction for second moment
    s_corrected = s_new / (1 - (beta2 ** t))

    # 5. Update the variable
    # We break this into parts to keep line length under 79 chars
    denom = np.sqrt(s_corrected) + epsilon
    var_new = var - (alpha * (v_corrected / denom))

    return var_new, v_new, s_new
