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

        # Weights vector initialized with random normal distribution
        # Shaped as (1, nx) to handle matrix multiplication with input X
        self.W = np.random.randn(1, nx)
        # Bias initialized to 0
        self.b = 0
        # Activated output (prediction) initialized to 0
        self.A = 0
