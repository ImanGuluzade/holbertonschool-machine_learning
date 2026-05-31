#!/usr/bin/env python3
"""
Module containing the convolve function for multiple convolution kernels.
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.

    Parameters:
    images: numpy.ndarray with shape (m, h, w, c) containing multiple images
    kernels: numpy.ndarray with shape (kh, kw, c, nc) containing the kernels
    padding: string ('same', 'valid') or tuple of (ph, pw)
    stride: tuple of (sh, sw) containing the strides for height and width

    Returns:
    A numpy.ndarray containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    # Parse and compute spatial padding boundaries accurately
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif isinstance(padding, tuple):
        ph, pw = padding

    # Calculate final matrix dimensions using step divisions
    h_new = int((h + (2 * ph) - kh) / sh) + 1
    w_new = int((w + (2 * pw) - kw) / sw) + 1

    # Pad only the spatial dimensions (height and width)
    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Pre-allocate output matrix to store all feature map outputs
    output = np.zeros((m, h_new, w_new, nc))

    # Loop through every feature map index filter layer (3 loops max total)
    for k in range(nc):
        kernel = kernels[:, :, :, k]
        for i in range(h_new):
            for j in range(w_new):
                v_start = i * sh
                v_end = v_start + kh
                h_start = j * sw
                h_end = h_start + kw

                # Extract multi-channel sliding image window volume slice
                slice_window = images_padded[:, v_start:v_end, h_start:h_end]

                # Multiply, sum across spatial axes + channel, store in map
                output[:, i, j, k] = np.sum(
                    slice_window * kernel,
                    axis=(1, 2, 3)
                )

    return output
