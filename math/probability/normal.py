#!/usr/bin/env python3
"""
Contains the Normal class
"""


class Normal:
    """
    Class that represents a normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes the normal distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean
            self.mean = float(sum(data) / len(data))

            # Calculate variance
            sum_diff_sq = 0
            for x in data:
                sum_diff_sq += (x - self.mean) ** 2
            variance = sum_diff_sq / len(data)

            # Standard deviation
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        """
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        """
        # Constants
        pi = 3.1415926536
        e = 2.7182818285

        # Parts of the formula
        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2
        coefficient = 1 / (self.stddev * (2 * pi) ** 0.5)

        return coefficient * (e ** exponent)
