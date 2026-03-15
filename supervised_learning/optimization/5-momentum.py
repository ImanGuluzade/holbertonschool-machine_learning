#!/usr/bin/env python3
"""
Module to update variables using Gradient Descent with Momentum
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum
    optimization algorithm

    Args:
        alpha: the learning rate
        beta1: the momentum weight
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        v: the previous first moment (velocity) of var

    Returns:
        The updated variable and the new moment, respectively
    """
    # 1. Update the velocity (first moment)
    # v_new = beta1 * v_prev + (1 - beta1) * gradient
    new_v = (beta1 * v) + ((1 - beta1) * grad)

    # 2. Update the variable
    # var_new = var_old - alpha * v_new
    new_var = var - (alpha * new_v)

    return new_var, new_v
