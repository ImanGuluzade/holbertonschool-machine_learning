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
        self.generate_features()

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
        Swaps internal MaxPooling2D layers for AveragePooling2D layers.
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        x = vgg.input
        outputs = []
        layer_dict = {}

        for layer in vgg.layers:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )(x)
            elif isinstance(layer, tf.keras.layers.InputLayer):
                continue
            else:
                x = layer(x)

            layer_dict[layer.name] = x

        for name in self.style_layers:
            outputs.append(layer_dict[name])
        outputs.append(layer_dict[self.content_layer])

        model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

        for layer in model.layers:
            layer.trainable = False

        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of a specific layer output tensor.
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or len(
                input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape

        num_locations = tf.cast(h * w, tf.float32)
        features = tf.reshape(input_layer, (h * w, c))

        gram = tf.matmul(features, features, transpose_a=True)
        gram_normalized = gram / num_locations

        return tf.expand_dims(gram_normalized, axis=0)

    def generate_features(self):
        """
        Extracts features used to calculate style and content cost.
        """
        preprocessed_style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255.0
        )
        preprocessed_content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255.0
        )

        style_outputs = self.model(preprocessed_style)
        content_outputs = self.model(preprocessed_content)

        self.gram_style_features = [
            self.gram_matrix(style_layer) for style_layer in style_outputs[:-1]
        ]

        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer.
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or len(
                style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        _, h, w, c = style_output.shape

        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
                gram_target.shape != (1, c, c):
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {c}, {c}]"
            )

        gram_style = self.gram_matrix(style_output)

        layer_loss = tf.reduce_sum(tf.square(gram_style - gram_target))
        normalization_factor = tf.cast(c ** 2, tf.float32)

        return layer_loss / normalization_factor

    def style_cost(self, style_outputs):
        """
        Calculates the overall style cost for the generated image.
        """
        num_layers = len(self.style_layers)

        if not isinstance(style_outputs, list) or \
                len(style_outputs) != num_layers:
            raise TypeError(
                f"style_outputs must be a list with a length of {num_layers}"
            )

        weight = 1.0 / float(num_layers)
        total_style_cost = 0.0

        for i in range(num_layers):
            layer_cost = self.layer_style_cost(
                style_outputs[i],
                self.gram_style_features[i]
            )
            total_style_cost += weight * layer_cost

        return tf.convert_to_tensor(total_style_cost, dtype=tf.float32)
