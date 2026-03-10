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

            # Input size: nx for the first layer, else size of previous layer
            if l == 0:
                layer_input = nx
            else:
                layer_input = layers[l - 1]

            # He et al. initialization: W = randn * sqrt(2 / n_prev)
            # Using '*' clearly for multiplication
            he_init = np.sqrt(2 / layer_input)
            self.weights["W{}".format(l + 1)] = (
                np.random.randn(layers[l], layer_input) * he_init
            )
            self.weights["b{}".format(l + 1)] = np.zeros((layers[l], 1))
