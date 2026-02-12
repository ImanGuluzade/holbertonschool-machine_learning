#!/usr/bin/env python3
"""Main file to test Poisson class without imports"""

Poisson = __import__('poisson').Poisson

# Example data for estimating lambtha
data = [4, 5, 5, 6, 5, 4, 5, 6, 5, 5]

# Create Poisson instance from data
p1 = Poisson(data)
print('Lambtha from data:', p1.lambtha)
print('P(9):', p1.pmf(9))

# Create Poisson instance with given lambtha
p2 = Poisson(lambtha=5)
print('Lambtha given:', p2.lambtha)
print('P(9):', p2.pmf(9))
