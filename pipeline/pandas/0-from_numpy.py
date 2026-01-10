#!/usr/bin/env python3
"""
This module contains the function from_numpy that
creates a pandas DataFrame from a numpy ndarray.
"""
import pandas as pd


def from_numpy(array):
    """
    Create a pandas DataFrame from a numpy ndarray.

    Parameters
    ----------
    array : numpy.ndarray
        The input array to convert.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with columns labeled A, B, C...
    """
    # Create column names: A, B, C, ...
    columns = [chr(65 + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)
