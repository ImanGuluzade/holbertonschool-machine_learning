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

            mean = sum(data) / len(data)
            sum_diff_sq = 0
            for x in data:
                sum_diff_sq += (x - mean) ** 2
            variance = sum_diff_sq / len(data)

            p_initial = 1 - (variance / mean)
            self.n = int(round(mean / p_initial))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        # nCr = n! / (k! * (n-k)!)
        n_fact = self.factorial(self.n)
        k_fact = self.factorial(k)
        nk_fact = self.factorial(self.n - k)
        ncr = n_fact / (k_fact * nk_fact)

        # PMF = nCr * p^k * (1-p)^(n-k)
        prob = ncr * (self.p ** k) * ((1 - self.p) ** (self.n - k))

        return prob

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes
        """
        k = int(k)
        if k < 0:
            return 0
        if k > self.n:
            return 1

        cumulative = 0
        for i in range(k + 1):
            cumulative += self.pmf(i)

        return cumulative

    def factorial(self, num):
        """
        Helper method to calculate factorial
        """
        f = 1
        for i in range(1, num + 1):
            f *= i
        return f
