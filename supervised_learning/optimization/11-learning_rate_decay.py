#!/usr/bin/env python3
"""
Module to update learning rate using inverse time decay
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy

    Args:
        alpha: the original learning rate
        decay_rate: the weight used to determine the rate of decay
        global_step: the number of passes of gradient descent elapsed
        decay_step: the number of passes before alpha is decayed further

    Returns:
        The updated value for alpha
    """
    # Stepwise decay: use floor division to keep the step constant
    # until decay_step is reached
    step = global_step // decay_step
    alpha_decayed = alpha / (1 + decay_rate * step)

    return alpha_decayed
