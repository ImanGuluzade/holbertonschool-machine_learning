#!/usr/bin/env python3
"""
Main file to test the Normal CDF method
"""

import numpy as np
Normal = __import__('normal').Normal

np.random.seed(0)
# Generate 100 samples with mean 70 and stddev 10
data = np.random.normal(70, 10, 100).tolist()
n1 = Normal(data)
# Testing CDF from data-calculated parameters
print('PHI(90):', n1.cdf(90))

n2 = Normal(mean=70, stddev=10)
# Testing CDF from manually set parameters
print('PHI(90):', n2.cdf(90))
