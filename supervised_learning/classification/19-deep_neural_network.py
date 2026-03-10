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
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                prev = nx
            else:
                prev = layers[i - 1]

            w_key = "W{}".format(i + 1)
            b_key = "b{}".format(i + 1)
            he_std = np.sqrt(2 / prev)

            self.__weights[w_key] = np.random.randn(layers[i], prev) * he_std
            self.__weights[b_key] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for the number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for the intermediary values dictionary"""
        return self.__cache

    @property
    def weights(self):
        """Getter for the weights and biases dictionary"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates forward propagation"""
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            prev_A = self.__cache["A{}".format(i - 1)]
            W = self.__weights["W{}".format(i)]
            b = self.__weights["b{}".format(i)]
            Z = np.dot(W, prev_A) + b
            self.__cache["A{}".format(i)] = 1 / (1 + np.exp(-Z))
        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Y: correct labels (1, m)
        A: activated output (1, m)
        Returns: the cost
        """
        m = Y.shape[1]
        # Use 1.0000001 - A to avoid division by zero
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = (1 / m) * np.sum(loss)
        return cost
