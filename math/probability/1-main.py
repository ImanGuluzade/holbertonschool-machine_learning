#!/usr/bin/env python3
"""Test Poisson PMF"""

Poisson = __import__('poisson').Poisson

if __name__ == "__main__":
    # Using sample data
    data = [4, 5, 2, 5, 6, 4, 5, 5, 6, 4]
    p1 = Poisson(data)
    print(p1.pmf(2))

    # Using given lambtha
    p2 = Poisson(lambtha=5)
    print(p2.pmf(9))
