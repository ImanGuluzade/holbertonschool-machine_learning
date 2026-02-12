#!/usr/bin/env python3
"""Test Poisson class PMF"""

Poisson = __import__('poisson').Poisson

if __name__ == "__main__":
    # Test Poisson using given data
    data = [4, 5, 2, 5, 6, 4, 5, 5, 6, 4]  # sample data
    p1 = Poisson(data)
    print('P(9):', p1.pmf(9))

    # Test Poisson using given lambtha
    p2 = Poisson(lambtha=5)
    print('P(9):', p2.pmf(9))
