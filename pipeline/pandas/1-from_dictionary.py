#!/usr/bin/env python3
"""
This module creates a pandas DataFrame from a dictionary.
"""

import pandas as pd

# Create the dictionary
data = {
    "First": [0.0, 0.5, 1.0, 1.5],
    "Second": ["one", "two", "three", "four"]
}

# Create the DataFrame with row labels A, B, C, D
df = pd.DataFrame(data, index=["A", "B", "C", "D"])
