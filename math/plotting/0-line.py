#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def line():
    """Plot y = x^3 as a solid red line from x=0 to 10."""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(np.arange(0, 11), y, 'r-')  # solid red line
    plt.xlim(0, 10)
    plt.show()
