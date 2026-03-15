#!/usr/bin/env python3
"""Module to convert labels to one-hot matrix using Keras"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Args:
        labels: the label vector to be converted.
        classes: the number of classes. Defaults to None.

    Returns:
        The one-hot matrix.
    """
    # K.utils.to_categorical handles the conversion to one-hot encoding
    # The last dimension will automatically be the number of classes
    return K.utils.to_categorical(labels, num_classes=classes)
