#!/usr/bin/env python3
"""
Contains the Exponential class
"""


class Exponential:
    """
    Class that represents an exponential distribution
    """

    def __init__(self, data=None, lambtha=1.0):
        """
        Initializes the exponential distribution
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Lambtha is the inverse of the mean
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period
        """
        if x < 0:
            return 0

        # Euler's number constant
        e = 2.7182818285

        # Formula: l * e^(-l * x)
        return self.lambtha * (e ** (-self.lambtha * x))
