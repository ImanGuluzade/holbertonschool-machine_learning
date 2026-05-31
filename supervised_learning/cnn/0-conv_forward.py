#!/usr/bin/env python3
"""
Module containing the conv_forward function for custom CNN layers.
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network

    Parameters:
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
    b: numpy.ndarray of shape (1, 1, 1, c_new)
    activation: activation function applied to the convolution
    padding: string that is either 'same' or 'valid'
    stride: tuple of (sh, sw) containing the strides

    Returns:
    The activated output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # Determine padding dimensions
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == "valid":
        ph, pw = 0, 0

    # Calculate spatial output feature map dimensions
    h_new = int((h_prev - kh + 2 * ph) / sh) + 1
    w_new = int((w_prev - kw + 2 * pw) / sw) + 1

    # Apply padding to the inputs
    A_padded = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Pre-allocate output matrix
    output = np.zeros((m, h_new, w_new, c_new))

    # Perform slice calculation multiplication over sliding window locations
    for h in range(h_new):
        for w in range(w_new):
            # Calculate input slice matrix window borders
            v_start = h * sh
            v_end = v_start + kh
            h_start = w * sw
            h_end = h_start + kw

            # Extract window from padded inputs
            slice_A = A_padded[:, v_start:v_end, h_start:h_end, :, np.newaxis]

            # Compute matrix product across channels element-wise
            # Element-wise product over W -> sum over axes (height, width, ch)
            conv_sum = np.sum(slice_A * W, axis=(1, 2, 3))
            output[:, h, w, :] = conv_sum

    # Add the bias tensor parameters and apply the activation function wrapper
    return activation(output + b)
