#!/usr/bin/env python3
"""
This module contains the function array that
selects the last 10 rows of High and Close columns
from a DataFrame and converts them to a numpy array.
"""

import pandas as pd
import numpy as np


def array(df):
    """
    Select last 10 rows of High and Close columns
    and convert to a numpy.ndarray.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'High' and 'Close' columns.

    Returns
    -------
    numpy.ndarray
        Array of shape (10, 2) with the last 10 High and Close values.
    """
    # Select last 10 rows of High and Close
    last10 = df[['High', 'Close']].tail(10)
    
    # Convert to numpy array
    return last10.to_numpy()
