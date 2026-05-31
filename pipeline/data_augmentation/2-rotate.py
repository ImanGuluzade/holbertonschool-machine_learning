#!/usr/bin/env python3
"""
Module containing the rotate_image function for data augmentation.
"""
import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image 90 degrees counter-clockwise.

    Parameters:
    image: a 3D tf.Tensor containing the image to rotate

    Returns:
    The rotated image tensor
    """
    return tf.image.rot90(image, k=1)
