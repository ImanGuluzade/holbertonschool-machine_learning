#!/usr/bin/env python3
"""
Module to create an RMSProp optimizer in TensorFlow
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in TensorFlow

    Args:
        alpha: the learning rate
        beta2: the RMSProp weight (discounting factor)
        epsilon: small number to avoid division by zero

    Returns:
        The optimizer
    """
    # rho is the TensorFlow equivalent of beta2 in RMSProp
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
