#!/usr/bin/env python3
"""
Module to calculate weighted moving average with bias correction
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set

    Args:
        data: list of data to calculate the moving average of
        beta: the weight used for the moving average

    Returns:
        A list containing the moving averages of data
    """
    moving_averages = []
    v = 0

    for i in range(len(data)):
        # Calculate exponentially weighted average
        v = beta * v + (1 - beta) * data[i]
        
        # Apply bias correction: v / (1 - beta^t)
        # Note: i + 1 because the formula uses t starting at 1
        bias_correction = 1 - (beta ** (i + 1))
        moving_averages.append(v / bias_correction)

    return moving_averages
