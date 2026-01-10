#!/usr/bin/env python3
import pandas as pd


def from_numpy(array):
    # Create column names: A, B, C, ...
    columns = [chr(65 + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)
