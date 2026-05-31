#!/usr/bin/env python3
"""
Module containing pool_backward function for custom CNN layers.
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network.

    Parameters:
    dA: numpy.ndarray of shape (m, h_new, w_new, c_new) containing partial
        derivatives with respect to the output of the pooling layer
    A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
            the output of the previous layer
    kernel_shape: tuple of (kh, kw) containing the size of the kernel
    stride: tuple of (sh, sw) containing the strides
    mode: string containing either 'max' or 'avg'

    Returns:
    dA_prev: the partial derivatives with respect to the previous layer
    """
    m, h_new, w_new, c_new = dA.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Initialize previous layer gradient matrix tracking map
    dA_prev = np.zeros_like(A_prev)

    # Distribute tracking derivatives back over sliding windows
    for h in range(h_new):
        for w in range(w_new):
            # Calculate input slice matrix window boundaries
            v_start = h * sh
            v_end = v_start + kh
            h_start = w * sw
            h_end = h_start + kw

            # Route gradients based on pooling operation mode rules
            if mode == 'max':
                for i in range(m):
                    for c in range(c_new):
                        # Extract the slice window from forward pass
                        slice_A = A_prev[i, v_start:v_end, h_start:h_end, c]

                        # Create binary mask matching position of maximum value
                        mask = (slice_A == np.max(slice_A))

                        # Accumulate the incoming gradient via the mask
                        dA_prev[i, v_start:v_end, h_start:h_end, c] += (
                            mask * dA[i, h, w, c]
                        )

            elif mode == 'avg':
                # Distribute average gradient split uniformly across window
                avg_dA = dA[:, h, w, :, np.newaxis, np.newaxis] / (kh * kw)

                # Reshape and transpose to match (m, kh, kw, c) layout
                avg_dA = np.transpose(avg_dA, (0, 2, 3, 1))

                # Accumulate distributed gradients inside target window
                dA_prev[:, v_start:v_end, h_start:h_end, :] += avg_dA

    return dA_prev
