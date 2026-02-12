#!/usr/bin/env python3
"""Test Poisson class"""

import numpy as np
from poisson import Poisson

if __name__ == "__main__":
    # Generate sample data
    np.random.seed(0)
    data = np.random.poisson(5., 100).tolist()

    # Test Poisson using data
    p1 = Poisson(data)
    print('Lambtha from data:', p1.lambtha)

    # Test Poisson using given lambtha
    p2 = Poisson(lambtha=5)
    print('Lambtha from lambtha:', p2.lambtha)
