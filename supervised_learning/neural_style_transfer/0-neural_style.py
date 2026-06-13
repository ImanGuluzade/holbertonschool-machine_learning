#!/usr/bin/env python3
"""
Defines the NST class to perform tasks for Neural Style Transfer.
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    NST class components for managing Neural Style Transfer configurations.
    """
    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes the NST instance parameters and checks types.

        Parameters:
            style_image: image used as a style reference (np.ndarray).
            content_image: image used as a content reference (np.ndarray).
            alpha: float representing the weight for content cost.
            beta: float representing the weight for style cost.
        """
        if (not isinstance(style_image, np.ndarray) or
                len(style_image.shape) != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray) or
                len(content_image.shape) != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is exactly 512 pixels.

        Parameters:
            image: np.ndarray of shape (h, w, 3) to scale.

        Returns:
            The scaled image as an eager tf.Tensor of shape (1, h_new, w_new, 3)
        """
        if (not isinstance(image, np.ndarray) or
                len(image.shape) != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape

        if h > w:
            h_new = 512
            w_new = int(round((w * 512) / h))
        else:
            w_new = 512
            h_new = int(round((h * 512) / w))

        # Add batch dimension required for tf.image.resize
        img_expanded = tf.expand_dims(image, axis=0)

        # Resize utilizing explicit bicubic interpolation
        resized_img = tf.image.resize(
            img_expanded, (h_new, w_new),
            method=tf.image.ResizeMethod.BICUBIC
        )

        # Normalize pixel values from range [0, 255] down to [0, 1]
        scaled_img = resized_img / 255.0

        # Enforce boundary limitations to ensure values stay strict inside [0,1]
        scaled_img = tf.clip_by_value(scaled_img, 0.0, 1.0)

        return scaled_img
