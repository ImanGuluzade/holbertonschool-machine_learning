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

            # Calculate variance: sum((x - mean)^2) / n
            sum_diff_sq = 0
            for x in data:
                sum_diff_sq += (x - self.mean) ** 2
            variance = sum_diff_sq / len(data)

            # Standard deviation is the square root of variance
            self.stddev = float(variance ** 0.5)
