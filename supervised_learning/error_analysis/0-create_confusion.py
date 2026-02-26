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
    # Using matrix multiplication for efficiency with one-hot vectors
    # labels.T (classes, m) x logits (m, classes) = (classes, classes)
    return np.matmul(labels.T, logits)
