#!/usr/bin/env python3
"""
Module containing the identity_block function for a ResNet architecture.
"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in Deep Residual Learning
    for Image Recognition (2015).

    Parameters:
    A_prev: tensor output from the previous layer
    filters: tuple or list containing F11, F3, F12, respectively:
             F11 is the number of filters in the first 1x1 convolution
             F3 is the number of filters in the 3x3 convolution
             F12 is the number of filters in the second 1x1 convolution

    Returns:
    The activated output of the identity block
    """
    F11, F3, F12 = filters

    # Initialize weights using He normal distribution with seed 0
    initializer = K.initializers.HeNormal(seed=0)

    # --- First Layer Components ---
    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(A_prev)
    bn1 = K.layers.BatchNormalization(axis=-1)(conv1)
    act1 = K.layers.Activation('relu')(bn1)

    # --- Second Layer Components ---
    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=initializer
    )(act1)
    bn2 = K.layers.BatchNormalization(axis=-1)(conv2)
    act2 = K.layers.Activation('relu')(bn2)

    # --- Third Layer Components ---
    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=initializer
    )(act2)
    bn3 = K.layers.BatchNormalization(axis=-1)(conv3)

    # --- Shortcut Connection ---
    merged = K.layers.Add()([bn3, A_prev])
    output = K.layers.Activation('relu')(merged)

    return output
