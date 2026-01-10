#!/usr/bin/env python3
"""
This module contains the function slice that
extracts certain columns and selects every 60th row.
"""


def slice(df):
    """
    Extract High, Low, Close, Volume_(BTC) columns
    and select every 60th row.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the required columns.

    Returns
    -------
    pandas.DataFrame
        Sliced DataFrame.
    """
    # Extract columns
    cols = df[['High', 'Low', 'Close', 'Volume_(BTC)']]
    
    # Select every 60th row
    return cols.iloc[::60]
