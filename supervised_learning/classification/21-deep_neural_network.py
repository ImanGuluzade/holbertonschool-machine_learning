#!/usr/bin/env python3
"""Module for Deep Neural Network"""
import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network performing binary classification"""

    def __init__(self, nx, layers):
        """Constructor: Loop 1 of 3"""
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
            prev = nx if i == 0 else layers[i - 1]
            w_k, b_k = "W" + str(i + 1), "b" + str(i + 1)
            he = np.sqrt(2 / prev)
            self.__weights[w_k] = np.random.randn(layers[i], prev) * he
            self.__weights[b_k] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """L getter"""
        return self.__L

    @property
    def cache(self):
        """cache getter"""
        return self.__cache

    @property
    def weights(self):
        """weights getter"""
        return self.__weights

    def forward_prop(self, X):
        """Forward Prop: Loop 2 of 3"""
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            A_p = self.__cache["A" + str(i - 1)]
            W = self.__weights["W" + str(i)]
            b = self.__weights["b" + str(i)]
            Z = np.dot(W, A_p) + b
            self.__cache["A" + str(i)] = 1 / (1 + np.exp(-Z))
        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Cost calculation (No loops)"""
        m = Y.shape[1]
        c = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return c

    def evaluate(self, X, Y):
        """Evaluation (No loops)"""
        A, _ = self.forward_prop(X)
        return np.where(A >= 0.5, 1, 0), self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Gradient Descent: Loop 3 of 3"""
        m = Y.shape[1]
        dz = cache["A" + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A_p = cache["A" + str(i - 1)]
            W_k, b_k = "W" + str(i), "b" + str(i)
            W = self.__weights[W_k]
            dw = (1 / m) * np.dot(dz, A_p.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            if i > 1:
                dz = np.dot(W.T, dz) * (A_p * (1 - A_p))
            self.__weights[W_k] -= alpha * dw
            self.__weights[b_k] -= alpha * db
