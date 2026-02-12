#!/usr/bin/env python3
"""Test Poisson PMF"""

Poisson = __import__('poisson').Poisson

# Use the same seed values as in exercise
data = [5, 4, 5, 6, 5, 5, 5, 4, 5, 5,
        6, 4, 5, 5, 5, 5, 4, 5, 6, 5,
        4, 5, 5, 5, 5, 6, 4, 5, 5, 5,
        5, 4, 5, 6, 5, 5, 5, 5, 4, 5,
        5, 5, 5, 6, 4, 5, 5, 5, 5, 6,
        4, 5, 5, 5, 5, 4, 5, 6, 5, 5,
        5, 5, 4, 5, 5, 5, 5, 6, 4, 5,
        5, 5, 5, 4, 5, 6, 5, 5, 5, 5,
        4, 5, 5, 5, 5, 6, 4, 5, 5, 5,
        5, 5, 5, 4, 5, 5, 5, 5, 6, 4]

p1 = Poisson(data)
print("P(9):", p1.pmf(9))

p2 = Poisson(lambtha=5)
print("P(9):", p2.pmf(9))
