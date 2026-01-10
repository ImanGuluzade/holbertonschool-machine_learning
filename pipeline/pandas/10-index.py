#!/usr/bin/env python3
"""
This module contains the function index that
sets the Timestamp column as the DataFrame index.
"""


def index(df):
    """
    Set the Timestamp column as the index of the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a 'Timestamp' column.

    Returns
    -------
    pandas.DataFrame
        DataFrame with Timestamp set as the index.
    """
    return df.set_index('Timestamp')
