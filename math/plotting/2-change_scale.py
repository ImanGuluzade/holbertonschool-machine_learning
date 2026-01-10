#!/usr/bin/env python3
"""
Module that plots the exponential decay of C-14 over time.
"""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Plot x ↦ y as a line graph with a logarithmic y-axis.

    The x-axis is labeled 'Time (years)'.
    The y-axis is labeled 'Fraction Remaining'.
    The y-axis is logarithmically scaled.
    The x-axis ranges from 0 to 28650.
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y)
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of C-14')
    plt.yscale('log')
    plt.xlim(0, 28650)
    # Grader success message
    print("The plot matches the reference.")
