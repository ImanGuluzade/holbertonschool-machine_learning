#!/usr/bin/env python3
"""
Module containing the convolve_grayscale_valid function.
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Parameters:
    images: numpy.ndarray with shape (m, h, w) containing multiple images
    kernel: numpy.ndarray with shape (kh, kw) containing the convolution kernel

    Returns:
    A numpy.ndarray with shape (m, h - kh + 1, w - kw + 1) containing the
    convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate spatial dimensions of the output feature maps
    h_new = h - kh + 1
    w_new = w - kw + 1

    # Pre-allocate output matrix to store processed activation maps
    output = np.zeros((m, h_new, w_new))

    # Perform convolution sliding window loops over output coordinates
    for i in range(h_new):
        for j in range(w_new):
            # Extract the localized window slice across all images at once
            slice_window = images[:, i:i + kh, j:j + kw]

            # Element-wise multiplication with kernel followed by spatial sums
            output[:, i, j] = np.sum(slice_window * kernel, axis=(1, 2))

    return output
