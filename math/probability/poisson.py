#!/usr/bin/env python3
"""Poisson distribution class without imports"""


class Poisson:
    """Represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculates the PMF for a given number of successes k"""
        k = int(k)
        if k < 0:
            return 0.0

        # Factorial manually
        fact = 1
        for i in range(1, k + 1):
            fact *= i

        # e^-l using Taylor series
        e_minus_l = 1.0
        term = 1.0
        for n in range(1, 100):
            term *= -self.lambtha / n
            e_minus_l += term

        return (self.lambtha ** k) * e_minus_l / fact
