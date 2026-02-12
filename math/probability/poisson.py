#!/usr/bin/env python3
"""
Poisson distribution class
"""


class Poisson:
    """
    Represents a Poisson distribution
    """

    def __init__(self, data=None, lambtha=1.0):
        """
        Initialize Poisson distribution
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
        # Convert k to integer
        k = int(k)

        # If k is out of range
        if k < 0:
            return 0

        # Euler's number constant
        e = 2.7182818285

        # Calculate k factorial (k!)
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        # Calculate PMF: (lambtha^k * e^-lambtha) / k!
        pdf_val = (self.lambtha ** k * (e ** -self.lambtha)) / factorial

        return pdf_val
