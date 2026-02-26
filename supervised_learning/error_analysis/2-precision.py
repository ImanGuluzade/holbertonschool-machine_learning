#!/usr/bin/env python3
"""
Module to calculate precision for each class in a confusion matrix
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix

    Args:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
                   represent correct labels and column indices represent
                   predicted labels

    Returns:
        A numpy.ndarray of shape (classes,) containing the precision
        of each class
    """
    # True Positives are the diagonal elements
    tp = np.diag(confusion)

    # Predicted instances for each class is the sum of each column
    # axis=0 sums down the columns
    predicted_total = np.sum(confusion, axis=0)

    # Precision = TP / (TP + FP)
    return tp / predicted_total
