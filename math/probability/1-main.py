#!/usr/bin/env python3
"""Test Poisson class PMF"""

import numpy as np
Poisson = __import__('poisson').Poisson

if __name__ == "__main__":
    # Generate sample data
    np.random.seed(0)
    data = np.random.poisson(5., 100).tolist()

    # Test Poisson using data
    p1 = Poisson(data)
    print('P(9):', p1.pmf(9))

    # Test Poisson using given lambtha
    p2 = Poisson(lambtha=5)
    print('P(9):', p2.pmf(9))
