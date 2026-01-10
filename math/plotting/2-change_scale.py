#!/usr/bin/env python3
"""
Exponential Decay plot for C-14
"""
import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """Plot C-14 exponential decay with a logarithmic y-axis."""
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Plot x vs y as a solid blue line
    plt.plot(x, y, 'b-')

    # Set axis labels and title
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of C-14")

    # Set y-axis to logarithmic scale
    plt.yscale("log")

    # Set x-axis limits
    plt.xlim(0, 28650)

    # Show the plot
    plt.show()
