#!/usr/bin/env python3
"""
Module containing the flip_image function for data augmentation.
"""
import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally.

    Parameters:
    image: a 3D tf.Tensor containing the image to flip

    Returns:
    The horizontally flipped image tensor
    """
    return tf.image.flip_left_right(image)
