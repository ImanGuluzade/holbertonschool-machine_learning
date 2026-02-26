#!/usr/bin/env python3
"""
Module to calculate sensitivity for each class in a confusion matrix
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix

    Args:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
                   represent correct labels and column indices represent
                   predicted labels

    Returns:
        A numpy.ndarray of shape (classes,) containing the sensitivity
        of each class
    """
    # True Positives are the diagonal elements
    tp = np.diag(confusion)

    # Actual instances for each class is the sum of each row
    actual_total = np.sum(confusion, axis=1)

    # Sensitivity = TP / (TP + FN)
    return tp / actual_total
