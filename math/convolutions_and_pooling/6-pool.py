#!/usr/bin/env python3
"""
Module containing the pool function for max and average pooling.
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs max or average pooling on images.

    Parameters:
    images: numpy.ndarray with shape (m, h, w, c) containing multiple images
    kernel_shape: tuple of (kh, kw) containing the pooling kernel dimensions
    stride: tuple of (sh, sw) containing the sliding stride steps
    mode: string indicating the pooling type ('max' or 'avg')

    Returns:
    A numpy.ndarray containing the pooled feature map output arrays.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate exact output matrix dimensions downsampled by stride steps
    h_new = int((h - kh) / sh) + 1
    w_new = int((w - kw) / sw) + 1

    # Pre-allocate output matrix preserving target channel layers
    output = np.zeros((m, h_new, w_new, c))

    # Loop over output coordinates spatial grids (strictly 2 loops max)
    for i in range(h_new):
        for j in range(w_new):
            v_start = i * sh
            v_end = v_start + kh
            h_start = j * sw
            h_end = h_start + kw

            # Slice the target spatial window block across all channels
            slice_window = images[:, v_start:v_end, h_start:h_end, :]

            # Select the appropriate mathematical summary projection logic
            if mode == 'max':
                output[:, i, j, :] = np.max(slice_window, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(slice_window, axis=(1, 2))

    return output
