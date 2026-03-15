#!/usr/bin/env python3
"""
Module to create a learning rate decay operation in TensorFlow
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation in tensorflow
    using inverse time decay

    Args:
        alpha: the original learning rate
        decay_rate: the weight used to determine the rate of decay
        decay_step: the number of passes of gradient descent that
                    should occur before alpha is decayed further

    Returns:
        The learning rate decay operation
    """
    # InverseTimeDecay implements alpha / (1 + decay_rate * step / decay_step)
    # staircase=True ensures the "stepwise" fashion
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )

    return lr_schedule
