#!/usr/bin/env python3
"""
Module defines a deep neural network performing binary classification
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Class that defines a deep neural network
    """

    def __init__(self, nx, layers):
        """
        Initializes the deep neural network
        nx: number of input features
        layers: list of nodes in each layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(self.L):
            if not isinstance(layers[l], int) or layers[l] <= 0:
                raise TypeError("layers must be a list of positive integers")

            # Calculate the number of inputs for the current layer
            if l == 0:
                prev_size = nx
            else:
                prev_size = layers[l - 1]

            # He et al. initialization
            # Using '*' for multiplication on a single line for clarity
            he_init = np.sqrt(2 / prev_size)
            w_key = "W" + str(l + 1)
            b_key = "b" + str(l + 1)

            self.weights[w_key] = np.random.randn(layers[l], prev_size) * he_init
            self.weights[b_key] = np.zeros((layers[l], 1))
