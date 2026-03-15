#!/usr/bin/env python3
"""
Module to calculate the moving average with bias correction
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set with bias correction

    Args:
        data: list of data to calculate the moving average of
        beta: the weight used for the moving average

    Returns:
        A list containing the moving averages of data
    """
    v = 0
    moving_averages = []

    for i in range(len(data)):
        # Calculate the moving average (momentum)
        v = (beta * v) + ((1 - beta) * data[i])

        # Apply bias correction for the current time step (i + 1)
        # Formula: v_corrected = v / (1 - beta^t)
        bias_correction = 1 - (beta ** (i + 1))
        v_corrected = v / bias_correction

        moving_averages.append(v_corrected)

    return moving_averages
