#!/usr/bin/env python3
"""
Module that defines a single neuron performing binary classification
"""
import numpy as np


class Neuron:
    """
    Class that defines a single neuron performing binary classification
    """
    def __init__(self, nx):
        """
        Initializes the neuron
        nx: number of input features
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
        """ Getter for weights vector """
        return self.__W

    @property
    def b(self):
        """ Getter for bias """
        return self.__b

    @property
    def A(self):
        """ Getter for activated output """
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        X: numpy.ndarray with shape (nx, m)
        Returns: the private attribute __A
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Y: numpy.ndarray with shape (1, m)
        A: numpy.ndarray with shape (1, m)
        Returns: the cost
        """
        m = Y.shape[1]
        loss = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = (1 / m) * np.sum(loss)
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions
        X: input data
        Y: correct labels
        Returns: prediction (1 or 0) and the cost
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        # Thresholding: 1 if A >= 0.5, else 0
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost
