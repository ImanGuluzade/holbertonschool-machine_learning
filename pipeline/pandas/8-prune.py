#!/usr/bin/env python3
"""
This module contains the function prune that
removes rows from a DataFrame where Close is NaN.
"""


def prune(df):
    """
    Remove any rows where the Close column has NaN values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a 'Close' column.

    Returns
    -------
    pandas.DataFrame
        DataFrame with rows containing NaN in Close removed.
    """
    return df[df['Close'].notna()]
