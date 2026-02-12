#!/usr/bin/env python3
"""
Contains the Binomial class
"""


class Binomial:
    """
    Class that represents a binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the binomial distribution
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Calculate mean (mu) and variance (sigma squared)
            mean = sum(data) / len(data)
            sum_diff_sq = 0
            for x in data:
                sum_diff_sq += (x - mean) ** 2
            variance = sum_diff_sq / len(data)

            # Using variance / mean = 1 - p
            # Therefore p = 1 - (variance / mean)
            p_initial = 1 - (variance / mean)

            # Calculate n and round to nearest integer
            self.n = int(round(mean / p_initial))

            # Recalculate p based on the rounded n
            self.p = float(mean / self.n)
