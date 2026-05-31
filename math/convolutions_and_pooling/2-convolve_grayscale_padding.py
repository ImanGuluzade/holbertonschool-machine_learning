#!/usr/bin/env python3
"""
Module containing the convolve_grayscale_padding function.
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding.

    Parameters:
    images: numpy.ndarray with shape (m, h, w) containing multiple images
    kernel: numpy.ndarray with shape (kh, kw) containing the convolution kernel
    padding: tuple of (ph, pw) containing padding heights and widths

    Returns:
    A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Calculate spatial dimensions of the custom padded output feature map
    h_new = h + (2 * ph) - kh + 1
    w_new = w + (2 * pw) - kw + 1

    # Apply the explicit custom zero-padding to the spatial dimensions
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    # Pre-allocate output matrix matching target dimensions
    output = np.zeros((m, h_new, w_new))

    # Perform convolution sliding window loops over target output coordinates
    for i in range(h_new):
        for j in range(w_new):
            # Extract localized slice windows from the padded image array
            slice_window = images_padded[:, i:i + kh, j:j + kw]

            # Multiply kernel over sliced regions and sum across spatial axes
            output[:, i, j] = np.sum(slice_window * kernel, axis=(1, 2))

    return output
