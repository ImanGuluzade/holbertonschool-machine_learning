# Probability Distributions

This folder contains exercises on probability distributions in Python.

## Exercises

### 0. Poisson Distribution

- Create a class `Poisson` that represents a Poisson distribution.
- The class constructor:
  - `__init__(self, data=None, lambtha=1.)`
  - `data`: a list of data points to estimate `lambtha`
  - `lambtha`: expected number of occurrences
- If `data` is provided:
  - Calculate `lambtha` from the data
  - Raise `TypeError` if data is not a list
  - Raise `ValueError` if data has fewer than 2 points
- If `data` is None:
  - Use the given `lambtha`
  - Raise `ValueError` if lambtha ≤ 0

## Usage

```python
from poisson import Poisson
import numpy as np

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
p = Poisson(data)
print(p.lambtha)
