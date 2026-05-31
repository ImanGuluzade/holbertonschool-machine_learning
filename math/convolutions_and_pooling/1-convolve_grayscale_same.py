#!/usr/bin/env python3
"""
Module containing the convolve_grayscale_same function.
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Parameters:
    images: numpy.ndarray with shape (m, h, w) containing multiple images
    kernel: numpy.ndarray with shape (kh, kw) containing the convolution kernel

    Returns:
    A numpy.ndarray with shape (m, h, w) containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate symmetric padding sizes for height and width
    ph = int(np.ceil((kh - 1) / 2))
    pw = int(np.ceil((kw - 1) / 2))

    # Apply constant zero padding to the spatial dimensions (axes 1 and 2)
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    # Pre-allocate output matrix matching the original dimensions
    output = np.zeros((m, h, w))

    # Perform convolution sliding window loops over target output coordinates
    for i in range(h):
        for j in range(w):
            # Extract localized slice windows from the padded image array
            slice_window = images_padded[:, i:i + kh, j:j + kw]

            # Multiply kernel over sliced regions and sum across spatial axes
            output[:, i, j] = np.sum(slice_window * kernel, axis=(1, 2))

    return output
