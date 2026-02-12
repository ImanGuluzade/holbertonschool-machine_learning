#!/usr/bin/env python3
"""
Poisson distribution module
"""


class Poisson:
    """
    Class that represents a Poisson distribution
    """

    def __init__(self, data=None, lambtha=1.0):
        """
        Initialization of the class
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
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes
        """
        k = int(k)
        if k < 0:
            return 0

        # Euler's number constant
        e = 2.7182818285

        # Calculate factorial of k
        fact = 1
        for i in range(1, k + 1):
            fact *= i

        # PMF Formula: (e^-L * L^k) / k!
        probability = ((e ** -self.lambtha) * (self.lambtha ** k)) / fact
        return probability

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes
        """
        k = int(k)
        if k < 0:
            return 0

        # Cumulative probability is the sum of PMFs from 0 to k
        cumulative = 0
        for i in range(k + 1):
            cumulative += self.pmf(i)

        return cumulative
