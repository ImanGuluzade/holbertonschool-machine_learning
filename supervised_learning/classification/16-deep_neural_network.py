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

        for i in range(len(layers)):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                prev_nodes = nx
            else:
                prev_nodes = layers[i - 1]

            # He et al. initialization
            # Using multiplication explicitly on one line
            w_name = "W" + str(i + 1)
            b_name = "b" + str(i + 1)
            
            # W = randn * sqrt(2 / input_features)
            self.weights[w_name] = np.random.randn(layers[i], prev_nodes) * np.sqrt(2 / prev_nodes)
            self.weights[b_name] = np.zeros((layers[i], 1))
