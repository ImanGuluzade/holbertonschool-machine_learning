#!/usr/bin/env python3
"""
Module to create a confusion matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix from one-hot encoded labels and logits

    Args:
        labels: one-hot numpy.ndarray of shape (m, classes)
                containing the correct labels
        logits: one-hot numpy.ndarray of shape (m, classes)
                containing the predicted labels

    Returns:
        A confusion numpy.ndarray of shape (classes, classes) with row indices
        representing correct labels and column indices representing predicted
    """
    # The dot product of labels.T and logits naturally sums up 
    # where the '1s' in both arrays overlap for each class pair.
    # labels.T: (classes, m) dot logits: (m, classes) -> (classes, classes)
    return np.matmul(labels.T, logits)
