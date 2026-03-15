#!/usr/bin/env python3
"""
Module to create a batch normalization layer in TensorFlow
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow

    Args:
        prev: the activated output of the previous layer
        n: the number of nodes in the layer to be created
        activation: the activation function to be used on the output
                    of the layer

    Returns:
        A tensor of the activated output for the layer
    """
    # Initialize the VarianceScaling as requested
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # Create the Dense layer. We use use_bias=False because the Beta
    # parameter in BatchNormalization serves as the bias/offset.
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )

    # Apply the dense layer to the previous layer's output
    z = dense(prev)

    # Create the Batch Normalization layer with the specified epsilon
    # gamma and beta are initialized to 1 and 0 by default in Keras
    batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-7)

    # Normalize the output (training=True ensures we calculate mean/var)
    z_norm = batch_norm(z, training=True)

    # Apply the activation function if it exists
    if activation is None:
        return z_norm

    return activation(z_norm)
