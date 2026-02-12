#!/usr/bin/env python3
"""
Main file to test Exponential class
"""

import numpy as np
Exponential = __import__('exponential').Exponential

np.random.seed(0)
# Generate data with scale 0.5 (which is 1/lambda, so lambda=2)
data = np.random.exponential(0.5, 100).tolist()
e1 = Exponential(data)
print('Lambtha:', e1.lambtha)

e2 = Exponential(lambtha=2)
print('Lambtha:', e2.lambtha)
