#!/usr/bin/env python3
"""
This module contains the function from_file that
loads data from a file into a pandas DataFrame.
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Load data from a file as a pandas DataFrame.

    Parameters
    ----------
    filename : str
        The path to the file to load.
    delimiter : str
        The column separator in the file.

    Returns
    -------
    pandas.DataFrame
        The loaded DataFrame.
    """
    return pd.read_csv(filename, delimiter=delimiter)
