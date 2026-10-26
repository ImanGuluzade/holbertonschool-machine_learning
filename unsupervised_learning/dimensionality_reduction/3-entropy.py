#!/usr/bin/env python3
"""Module that calculates the Shannon entropy and P affinities"""
import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to
    a data point

    Arguments:
        Di: numpy.ndarray of shape (n - 1,) containing the pairwise
            distances between a data point and all other points
            except itself
            n is the number of data points
        beta: numpy.ndarray of shape (1,) containing the beta value
            for the Gaussian distribution

    Returns:
        (Hi, Pi)
        Hi: the Shannon entropy of the points
        Pi: numpy.ndarray of shape (n - 1,) containing the P
            affinities of the points
    """
    num = np.exp(-Di * beta)
    Pi = num / np.sum(num)
    Hi = -np.sum(Pi * np.log2(Pi))
    return Hi, Pi
