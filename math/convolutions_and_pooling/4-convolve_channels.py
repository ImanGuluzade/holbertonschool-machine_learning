#!/usr/bin/env python3
"""
Module containing the convolve_channels function for multi-channel images.
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on multi-channel images using custom padding/strides.

    Parameters:
    images: numpy.ndarray with shape (m, h, w, c) containing multiple images
    kernel: numpy.ndarray with shape (kh, kw, c) containing the kernel
    padding: string ('same', 'valid') or tuple of (ph, pw)
    stride: tuple of (sh, sw) containing the strides for height and width

    Returns:
    A numpy.ndarray containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    # Parse and compute spatial padding boundaries safely
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int((kh - 1) / 2)
        pw = int((kw - 1) / 2)
    elif isinstance(padding, tuple):
        ph, pw = padding

    # Calculate final matrix dimensions using standard integer math
    h_new = int((h + (2 * ph) - kh) / sh) + 1
    w_new = int((w + (2 * pw) - kw) / sw) + 1

    # Pad only the spatial dimensions (height and width), leaving c untouched
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Pre-allocate output feature map matrix (3D array shape output)
    output = np.zeros((m, h_new, w_new))

    # Loop strictly over the output spatial grid boundaries (2 loops max)
    for i in range(h_new):
        for j in range(w_new):
            v_start = i * sh
            v_end = v_start + kh
            h_start = j * sw
            h_end = h_start + kw

            # Extract full depth-channel volumes across all batch elements
            slice_window = images_padded[:, v_start:v_end, h_start:h_end, :]

            # Compute sum over height, width, and channels simultaneously
            output[:, i, j] = np.sum(slice_window * kernel, axis=(1, 2, 3))

    return output
