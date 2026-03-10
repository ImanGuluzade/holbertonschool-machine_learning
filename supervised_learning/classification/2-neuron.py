#!/usr/bin/env python3
"""
Module that defines a single neuron performing forward propagation
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
        # Z = W . X + b
        Z = np.dot(self.__W, X) + self.__b
        # Sigmoid Activation Function
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
