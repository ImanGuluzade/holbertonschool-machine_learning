#!/usr/bin/env python3
"""Module for Multiclass Deep Neural Network"""
import numpy as np
import pickle
import os


class DeepNeuralNetwork:
    """Deep Neural Network performing multiclass classification"""

    def __init__(self, nx, layers):
        """Constructor for the neural network"""
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
            # He initialization
            he = np.sqrt(2 / prev)
            self.__weights[w_k] = np.random.randn(layers[i], prev) * he
            self.__weights[b_k] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter for number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates forward propagation with Softmax in the final layer
        X: input data of shape (nx, m)
        """
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            A_p = self.__cache["A" + str(i - 1)]
            W = self.__weights["W" + str(i)]
            b = self.__weights["b" + str(i)]
            Z = np.dot(W, A_p) + b

            if i < self.__L:
                # Sigmoid activation for hidden layers
                self.__cache["A" + str(i)] = 1 / (1 + np.exp(-Z))
            else:
                # Softmax activation for the output layer
                t = np.exp(Z)
                self.__cache["A" + str(i)] = t / np.sum(t, axis=0,
                                                        keepdims=True)
        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        Calculates the multiclass categorical cross-entropy cost
        Y: one-hot labels of shape (classes, m)
        A: activated output of shape (classes, m)
        """
        m = Y.shape[1]
        # Standard multiclass cost calculation
        cost = - (1 / m) * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the network predictions using argmax
        Returns: one-hot matrix and cost
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        # Find index of max probability for each example
        max_idx = np.argmax(A, axis=0)
        # Convert indices back to one-hot encoding
        oh = np.zeros(A.shape)
        oh[max_idx, np.arange(A.shape[1])] = 1
        return oh, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates gradient descent for multiclass labels"""
        m = Y.shape[1]
        dz = cache["A" + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A_p = cache["A" + str(i - 1)]
            W_k, b_k = "W" + str(i), "b" + str(i)
            W = self.__weights[W_k]

            dw = (1 / m) * np.dot(dz, A_p.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            if i > 1:
                # Gradient of Sigmoid
                dz = np.dot(W.T, dz) * (A_p * (1 - A_p))

            self.__weights[W_k] -= alpha * dw
            self.__weights[b_k] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """Trains the network over specified iterations"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        steps, costs = [], []
        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                c = self.cost(Y, A)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, c))
                if graph:
                    steps.append(i)
                    costs.append(c)
            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves instance to pickle format"""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads instance from pickle format"""
        if not os.path.exists(filename):
            return None
        with open(filename, 'rb') as f:
            return pickle.load(f)
