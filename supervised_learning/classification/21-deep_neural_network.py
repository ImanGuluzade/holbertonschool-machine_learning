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
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = (1 / m) * np.sum(loss)
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        Y: correct labels (1, m)
        cache: dictionary of intermediary values
        alpha: learning rate
        """
        m = Y.shape[1]
        # Initial dZ for the output layer (sigmoid)
        dZ = cache["A" + str(self.__L)] - Y

        for i in range(self.__L, 0, -1):
            # A_prev is the activation of the previous layer
            A_prev = cache["A" + str(i - 1)]
            W_key = "W" + str(i)
            b_key = "b" + str(i)
            W = self.__weights[W_key]

            # Calculate gradients
            dW = (1 / m) * np.dot(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

            # dZ for the next layer (moving backwards)
            # dZ_prev = (W_curr.T . dZ_curr) * (A_prev * (1 - A_prev))
            if i > 1:
                dZ = np.dot(W.T, dZ) * (A_prev * (1 - A_prev))

            # Update weights and biases
            self.__weights[W_key] = self.__weights[W_key] - (alpha * dW)
            self.__weights[b_key] = self.__weights[b_key] - (alpha * db)
