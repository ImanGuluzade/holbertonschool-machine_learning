#!/usr/bin/env python3
"""
Module to calculate specificity for each class in a confusion matrix
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix

    Args:
        confusion: numpy.ndarray of shape (classes, classes) where row indices
                   represent correct labels and column indices represent
                   predicted labels

    Returns:
        A numpy.ndarray of shape (classes,) containing the specificity
        of each class
    """
    # Total number of samples in the confusion matrix
    total = np.sum(confusion)

    # True Positives: diagonal elements
    tp = np.diag(confusion)

    # False Positives: sum of columns minus TP
    fp = np.sum(confusion, axis=0) - tp

    # False Negatives: sum of rows minus TP
    fn = np.sum(confusion, axis=1) - tp

    # True Negatives: Total samples - (TP + FP + FN)
    tn = total - (tp + fp + fn)

    # Specificity = TN / (TN + FP)
    return tn / (tn + fp)
