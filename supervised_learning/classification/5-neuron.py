#!/usr/bin/env python3
"""
Module that defines a single neuron performing gradient descent
"""
import numpy as np


class Neuron:
    """
    Class that defines a single neuron performing binary classification
    """
    def __init__(self, nx):
        """
        Initializes the neuron
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Getter for W """
        return self.__W

    @property
    def b(self):
        """ Getter for b """
        return self.__b

    @property
    def A(self):
        """ Getter for A """
        return self.__A

    def forward_prop(self, X):
        """ Calculates forward propagation """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """ Calculates cost using logistic regression """
        m = Y.shape[1]
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = (1 / m) * np.sum(loss)
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neuron predictions """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        X: input data
        Y: correct labels
        A: activated output
        alpha: learning rate
        Updates __W and __b
        """
        m = Y.shape[1]
        dz = A - Y
        # Compute gradients
        dw = (1 / m) * np.dot(dz, X.T)
        db = (1 / m) * np.sum(dz)
        # Update weights and bias
        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)
