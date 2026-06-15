#!/usr/bin/env python3
"""
Contains the NST class for Neural Style Transfer
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    NST class that performs tasks for Neural Style Transfer
    """
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block4_conv2'

    def __init__(self, style_image, content_image, alpha=1e-4, beta=1):
        """
        Initializes the NST class instance
        """
        if (not isinstance(style_image, tf.Tensor) and
                not isinstance(style_image, tf.Variable)):
            if len(style_image.shape) != 3 or style_image.shape[2] != 3:
                raise TypeError(
                    "style_image must be a numpy.ndarray with shape (h, w, 3)"
                )
        elif len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, tf.Tensor) and
                not isinstance(content_image, tf.Variable)):
            if len(content_image.shape) != 3 or content_image.shape[2] != 3:
                raise TypeError(
                    "content_image must be a numpy.ndarray with shape (h, w, 3)"
                )
        elif len(content_image.shape) != 3 or content_image.shape[2] != 3:
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
        Rescales an image to have pixel values between 0 and 1
        and centralizes its dimensions for VGG19 input
        """
        if (not isinstance(image, tf.Tensor) and
                not isinstance(image, tf.Variable)):
            if len(image.shape) != 3 or image.shape[2] != 3:
                raise TypeError(
                    "image must be a numpy.ndarray with shape (h, w, 3)"
                )
        elif len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        # 1. Resize the unscaled image first using bicubic interpolation
        image = tf.image.resize(image, [h_new, w_new], method='bicubic')
        
        # 2. Add the batch dimension
        image = tf.expand_dims(image, axis=0)
        
        # 3. Divide by 255.0 to normalize values between 0 and 1
        image = image / 255.0
        
        # 4. Clip values to prevent rounding overflow
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image

    def load_model(self):
        """
        Loads the VGG19 model pruned for Neural Style Transfer features
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in self.style_layers]
        outputs.append(vgg.get_layer(self.content_layer).output)

        custom_model = tf.keras.models.Model(vgg.input, outputs)
        self.model = custom_model

    @staticmethod
    def gram_matrix(input_tensor):
        """
        Calculates the Gram Matrix of a given feature map layer tensor
        """
        if (not isinstance(input_tensor, tf.Tensor) and
                not isinstance(input_tensor, tf.Variable)):
            raise TypeError("input_tensor must be a tensor")
        if len(input_tensor.shape) != 4:
            raise TypeError("input_tensor must be a tensor")

        channels = int(input_tensor.shape[-1])
        matrix = tf.reshape(input_tensor, [-1, channels])
        n = tf.shape(matrix)[0]
        gram = tf.matmul(matrix, matrix, transpose_a=True)
        return gram / tf.cast(n, tf.float32)

    def generate_features(self):
        """
        Extracts and stores the style and content feature maps
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
            self.gram_matrix(layer) for layer in style_outputs[:-1]
        ]
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single isolated layer
        """
        if (not isinstance(style_output, tf.Tensor) and
                not isinstance(style_output, tf.Variable)):
            raise TypeError("style_output must be a tensor")
        if len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor")

        c = style_output.shape[-1]
        if (not isinstance(gram_target, tf.Tensor) and
                not isinstance(gram_target, tf.Variable)):
            raise TypeError(
                "gram_target must be a tensor of shape ({}, {})".format(c, c)
            )
        if gram_target.shape != (c, c):
            raise TypeError(
                "gram_target must be a tensor of shape ({}, {})".format(c, c)
            )

        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """
        Calculates the overall style cost across all designated style layers
        """
        n_layers = len(self.style_layers)
        if not isinstance(style_outputs, list) or len(style_outputs) != n_layers:
            raise TypeError(
                "style_outputs must be a list of length {}".format(n_layers)
            )

        cost = 0.0
        weight = 1.0 / n_layers
        for i in range(n_layers):
            cost += weight * self.layer_style_cost(
                style_outputs[i],
                self.gram_style_features[i]
            )
        return cost

    def content_cost(self, content_output):
        """
        Calculates the content cost for the generated image
        """
        if (not isinstance(content_output, tf.Tensor) and
                not isinstance(content_output, tf.Variable)):
            raise TypeError("content_output must be a tensor")

        s = self.content_feature.shape
        if content_output.shape != s:
            raise TypeError(
                "content_output must be a tensor of shape {}".format(s)
            )

        return tf.reduce_mean(tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        """
        Calculates the weighted total loss sum
        """
        if (not isinstance(generated_image, tf.Tensor) and
                not isinstance(generated_image, tf.Variable)):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    self.content_image.shape
                )
            )
        if generated_image.shape != self.content_image.shape:
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    self.content_image.shape
                )
            )

        preprocessed = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255.0
        )
        outputs = self.model(preprocessed)

        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        J_style = self.style_cost(style_outputs)
        J_content = self.content_cost(content_output)

        J_total = (self.alpha * J_content) + (self.beta * J_style)
        return J_total, J_content, J_style

    def compute_grads(self, generated_image):
        """
        Calculates the gradients for the tf.Tensor generated image
        """
        if (not isinstance(generated_image, tf.Tensor) and
                not isinstance(generated_image, tf.Variable)):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    self.content_image.shape
                )
            )
        if generated_image.shape != self.content_image.shape:
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(
                    self.content_image.shape
                )
            )

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style = self.total_cost(generated_image)

        gradients = tape.gradient(J_total, generated_image)
        return gradients, J_total, J_content, J_style
