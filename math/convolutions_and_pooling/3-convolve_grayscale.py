#!/usr/bin/env python3
"""
Module containing the convolve_grayscale function.
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with custom padding and strides.

    Parameters:
    images: numpy.ndarray with shape (m, h, w) containing multiple images
    kernel: numpy.ndarray with shape (kh, kw) containing the convolution kernel
    padding: string ('same', 'valid') or tuple of (ph, pw)
    stride: tuple of (sh, sw) containing the strides for height and width

    Returns:
    A numpy.ndarray containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Parse and compute padding lengths based on input arguments
    if padding == 'valid':
        ph, pw = 0, 0
        h_new = int((h - kh) / sh) + 1
        w_new = int((w - kw) / sw) + 1
    elif padding == 'same':
        # Calculate output dimensions matching same padding stride rules
        h_new = int(np.ceil(h / sh))
        w_new = int(np.ceil(w / sw))
        ph = int(np.ceil(((h_new - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w_new - 1) * sw + kw - w) / 2))
    elif isinstance(padding, tuple):
        ph, pw = padding
        h_new = int((h + (2 * ph) - kh) / sh) + 1
        w_new = int((w + (2 * pw) - kw) / sw) + 1

    # Apply constant zero-padding symmetrically to spatial dimensions
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    # Pre-allocate output matrix matching target downsampled dimensions
    output = np.zeros((m, h_new, w_new))

    # Perform convolution sliding loops strictly over target coordinates
    for i in range(h_new):
        for j in range(w_new):
            # Calculate sliding window slice boundaries based on stride steps
            v_start = i * sh
            v_end = v_start + kh
            h_start = j * sw
            h_end = h_start + kw

            # Extract localized multi-image feature window slices
            slice_window = images_padded[:, v_start:v_end, h_start:h_end]

            # Multiply kernel over slices and sum across spatial dimensions
            output[:, i, j] = np.sum(slice_window * kernel, axis=(1, 2))

    return output
