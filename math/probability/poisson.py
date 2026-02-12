#!/usr/bin/env python3
"""Poisson distribution class"""


class Poisson:
    """Represents a Poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        if data is None:
            # Use provided lambtha
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            # Use data to calculate lambtha
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Calculates the PMF for a given number of successes k"""
        k = int(k)
        if k < 0:
            return 0

        # Calculate factorial manually
        fact = 1
        for i in range(1, k + 1):
            fact *= i

        # Calculate e^-λ manually using series expansion
        n_terms = 100  # more terms = more accurate
        e_minus_lambtha = 0
        for n in range(n_terms):
            term = (-self.lambtha) ** n
            term_fact = 1
            for i in range(1, n + 1):
                term_fact *= i
            e_minus_lambtha += term / (term_fact if term_fact != 0 else 1)

        # PMF formula
        return (self.lambtha ** k) * e_minus_lambtha / (fact if fact != 0 else 1)
