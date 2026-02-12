#!/usr/bin/env python3

import numpy as np
Poisson = __import__('poisson').Poisson

np.random.seed(0)
# Generate 100 samples from a Poisson distribution with lambda=5
data = np.random.poisson(5., 100).tolist()
p1 = Poisson(data)
print('P(9):', p1.pmf(9))

# Test with a manually set lambtha
p2 = Poisson(lambtha=5)
print('P(9):', p2.pmf(9))
