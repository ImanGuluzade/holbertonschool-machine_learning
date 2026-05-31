#!/usr/bin/env python3
"""
Module containing the conv_backward function for custom CNN layers.
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network.

    Parameters:
    dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing partial
        derivatives with respect to unactivated output of the conv layer
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
            the output of the previous layer
    W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing kernels
    b: numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    padding: string that is either 'same' or 'valid'
    stride: tuple of (sh, sw) containing the strides

    Returns:
    dA_prev: partial derivatives with respect to the previous layer
    dW: partial derivatives with respect to the kernels
    db: partial derivatives with respect to the biases
    """
    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    # Determine structural padding lengths used in forward step
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == "valid":
        ph, pw = 0, 0

    # Initialize gradients with target shape matrices
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Pad incoming activations and gradient placeholders symmetrically
    A_padded = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )
    dA_padded = np.pad(
        dA_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Accumulate mathematical slice derivatives across sliding matrix regions
    for h in range(h_new):
        for w in range(w_new):
            # Calculate input slice matrix window boundaries
            v_start = h * sh
            v_end = v_start + kh
            h_start = w * sw
            h_end = h_start + kw

            # Slice localized activation regions
            slice_A = A_padded[:, v_start:v_end, h_start:h_end, :]

            # Compute dW and dA_padded updates
            for c in range(c_new):
                dZ_val = dZ[:, h, w, c, np.newaxis, np.newaxis, np.newaxis]
                dW[:, :, :, c] += np.sum(slice_A * dZ_val, axis=0)

                # Keep shapes aligned explicitly for arbitrary batch sizes (m)
                dZ_broadcast = dZ[:, h, w, c, np.newaxis, np.newaxis, np.newaxis]
                dA_padded[:, v_start:v_end, h_start:h_end, :] += (
                    W[:, :, :, c] * dZ_broadcast
                )

    # Slice out original internal matrix spatial coordinates to discard padding
    if padding == "same" and ph > 0 and pw > 0:
        dA_prev = dA_padded[:, ph:-ph, pw:-pw, :]
    elif padding == "same" and ph > 0:
        dA_prev = dA_padded[:, ph:-ph, :, :]
    elif padding == "same" and pw > 0:
        dA_prev = dA_padded[:, :, pw:-pw, :]
    else:
        dA_prev = dA_padded

    return dA_prev, dW, db
