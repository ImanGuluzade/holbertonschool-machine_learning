#!/usr/bin/env python3
"""
Module containing the pool_forward function for custom pooling layers.
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network.

    Parameters:
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    kernel_shape: tuple of (kh, kw) containing the size of the kernel
    stride: tuple of (sh, sw) containing the strides
    mode: string containing either 'max' or 'avg'

    Returns:
    The output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate spatial output feature map dimensions
    h_new = int((h_prev - kh) / sh) + 1
    w_new = int((w_prev - kw) / sw) + 1

    # Pre-allocate output matrix matching target dimensions
    output = np.zeros((m, h_new, w_new, c_prev))

    # Iterate over sliding window dimensions
    for h in range(h_new):
        for w in range(w_new):
            # Calculate pooling spatial slice boundaries
            v_start = h * sh
            v_end = v_start + kh
            h_start = w * sw
            h_end = h_start + kw

            # Extract current feature map window block region
            slice_A = A_prev[:, v_start:v_end, h_start:h_end, :]

            # Apply the pool aggregation selector rule criteria
            if mode == 'max':
                output[:, h, w, :] = np.max(slice_A, axis=(1, 2))
            elif mode == 'avg':
                output[:, h, w, :] = np.mean(slice_A, axis=(1, 2))

    return output
