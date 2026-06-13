#!/usr/bin/env python3
"""
Contains the NST class for Neural Style Transfer
"""
import tensorflow as tf
import numpy as np


class NST:
    """
    Neural Style Transfer class
    """
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block4_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor
        """
        if not isinstance(style_image, np.ndarray) or \
           len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(content_image, np.ndarray) or \
           len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.alpha = alpha
        self.beta = beta
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)

        # Load the VGG19 model and extract targets
        self.model = self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales a numpy image to VGG19 target dimensions and normalization
        """
        if not isinstance(image, np.ndarray) or \
           len(image.shape) != 3 or image.shape[2] != 3:
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

        # Resize using bicubic interpolation
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.image.resize(
            image, [h_new, w_new],
            method=tf.image.ResizeMethod.BICUBIC
        )

        # Add batch dimension and scale to [0, 1]
        image = tf.expand_dims(image, axis=0)
        image = image / 255.0

        return image

    def load_model(self):
        """
        Creates the model used for Neural Style Transfer
        """
        vgg = tf.keras.applications.vGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        # Build custom outputs from target layers
        outputs = []
        for name in self.style_layers:
            outputs.append(vgg.get_layer(name).output)
        outputs.append(vgg.get_layer(self.content_layer).output)

        model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)
        return model

    @staticmethod
    def gram_matrix(input_tensor):
        """
        Calculates the Gram matrix for a specific layer output
        """
        if not isinstance(input_tensor, (tf.Tensor, tf.Variable)) or \
           len(input_tensor.shape) != 4:
            raise TypeError("input_tensor must be a tensor of rank 4")

        # Compute the outer product over spatial coordinates
        channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)

        return gram / tf.cast(n, tf.float32)

    def generate_features(self):
        """
        Extracts the target style features and content features from the images
        """
        # Preprocess using VGG19 expectations (scaled back to 0-255 internally)
        pre_style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255.0
        )
        pre_content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255.0
        )

        style_outputs = self.model(pre_style)
        content_outputs = self.model(pre_content)

        self.gram_style_features = [
            self.gram_matrix(layer) for layer in style_outputs[:-1]
        ]
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates style cost for a single layer
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
           len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[-1]
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
           gram_target.shape != (c, c):
            raise TypeError(
                f"gram_target must be a tensor of shape ({c}, {c})"
            )

        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """
        Calculates total style cost across all designated style layers
        """
        if not isinstance(style_outputs, list) or \
           len(style_outputs) != len(self.style_layers):
            raise TypeError(
                f"style_outputs must be a list of length "
                f"{len(self.style_layers)}"
            )

        weight = 1.0 / float(len(self.style_layers))
        costs = []

        for output, target in zip(style_outputs, self.gram_style_features):
            costs.append(self.layer_style_cost(output, target))

        return tf.reduce_sum(costs) * weight

    def content_cost(self, content_output):
        """
        Calculates content cost relative to the reference image target
        """
        if not isinstance(content_output, (tf.Tensor, tf.Variable)) or \
           content_output.shape != self.content_feature.shape:
            raise TypeError(
                f"content_output must be a tensor of shape "
                f"{self.content_feature.shape}"
            )

        return tf.reduce_mean(tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        """
        Calculates weighted sum of content and style losses
        """
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
           generated_image.shape != self.content_image.shape:
            raise TypeError(
                f"generated_image must be a tensor of shape "
                f"{self.content_image.shape}"
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
        Calculates the gradients for the generated image

        Args:
            generated_image: tf.Tensor of shape (1, nh, nw, 3)

        Returns:
            gradients: tf.Tensor containing gradients for generated image
            J_total: total cost for the generated image
            J_content: content cost for the generated image
            J_style: style cost for the generated image
        """
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
           generated_image.shape != self.content_image.shape:
            raise TypeError(
                f"generated_image must be a tensor of shape "
                f"{self.content_image.shape}"
            )

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style = self.total_cost(generated_image)

        gradients = tape.gradient(J_total, generated_image)

        return gradients, J_total, J_content, J_style
