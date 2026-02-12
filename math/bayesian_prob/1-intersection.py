#!/usr/bin/env python3
"""
Module to calculate the intersection for Bayesian probability
"""
import numpy as np


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining data with various
    hypothetical probabilities.

    Args:
        x: number of patients that develop severe side effects
        n: total number of patients observed
        P: 1D numpy.ndarray of hypothetical probabilities
        Pr: 1D numpy.ndarray of prior beliefs of P

    Returns:
        1D numpy.ndarray containing the intersection for each probability in P
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        err = "x must be an integer that is greater than or equal to 0"
        raise ValueError(err)
    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        err = "Pr must be a numpy.ndarray with the same shape as P"
        raise TypeError(err)

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate Likelihood: (nCr) * p^x * (1-p)^(n-x)
    n_fact = np.math.factorial(n)
    x_fact = np.math.factorial(x)
    nx_fact = np.math.factorial(n - x)
    combination = n_fact / (x_fact * nx_fact)

    likelihood = combination * (P ** x) * ((1 - P) ** (n - x))

    # Intersection = Likelihood * Prior
    return likelihood * Pr
