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
    # 1. Initialize the base Dense layer
    # We disable the built-in bias because beta in BatchNormalization
    # handles the offset, making the Dense bias redundant.
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense_layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )
    
    # 2. Perform the linear transformation: Z = WX
    z = dense_layer(prev)
    
    # 3. Apply Batch Normalization
    # Gamma and Beta are initialized to 1 and 0 by default in Keras.
    batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-7)
    z_norm = batch_norm(z)
    
    # 4. Apply the activation function
    if activation is None:
        return z_norm
    
    return activation(z_norm)
