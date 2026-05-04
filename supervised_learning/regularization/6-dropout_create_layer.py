#!/usr/bin/env python3
"""
Module to create a Keras layer with Dropout
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a neural network layer in TensorFlow using dropout

    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function for the new layer
        keep_prob: probability that a node will be kept
        training: boolean indicating whether the model is in training mode

    Returns:
        The output of the new layer
    """
    # Initialize weights using VarianceScaling
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg',
        distribution='normal'
    )

    # Create the Dense layer
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )

    # Dropout layer: rate is the probability of dropping (1 - keep_prob)
    dropout = tf.keras.layers.Dropout(rate=1 - keep_prob)

    # Apply the layers
    output = layer(prev)
    return dropout(output, training=training)
