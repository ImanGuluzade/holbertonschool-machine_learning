#!/usr/bin/env python3
"""Test Poisson class PMF"""

Poisson = __import__('poisson').Poisson

# Create a Poisson instance using data
data = [5, 6, 4, 5, 5, 4, 6, 5, 4, 5]
p1 = Poisson(data)
print("P(9):", p1.pmf(9))

# Create a Poisson instance using lambtha directly
p2 = Poisson(lambtha=5)
print("P(9):", p2.pmf(9))
