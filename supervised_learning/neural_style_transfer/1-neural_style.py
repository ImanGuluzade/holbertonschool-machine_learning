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
        """
        if not isinstance(style_image, np.ndarray) or len(
                style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(content_image, np.ndarray) or len(
                content_image.shape) != 3 or content_image.shape[2] != 3:
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
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is exactly 512 pixels.
        """
        if not isinstance(image, np.ndarray) or len(
                image.shape) != 3 or image.shape[2] != 3:
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

        img_expanded = tf.expand_dims(image, axis=0)

        resized_img = tf.image.resize(
            img_expanded, (h_new, w_new),
            method=tf.image.ResizeMethod.BICUBIC
        )

        scaled_img = resized_img / 255.0
        scaled_img = tf.clip_by_value(scaled_img, 0.0, 1.0)

        return scaled_img

    def load_model(self):
        """
        Creates and initializes the Keras model used to calculate cost.
        Uses a modified VGG19 base with average pooling layers.
        """
        # Load VGG19 directly forcing average pooling replacement structure
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )

        # Map clean layer references from original model definitions
        outputs = []
        for name in self.style_layers:
            outputs.append(vgg.get_layer(name).output)
        outputs.append(vgg.get_layer(self.content_layer).output)

        # Build clean structural functional model mapped to original inputs
        model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

        # Freeze parameter weights across model layers
        for layer in model.layers:
            layer.trainable = False

        self.model = model
