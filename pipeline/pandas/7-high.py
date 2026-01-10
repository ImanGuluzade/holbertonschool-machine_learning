#!/usr/bin/env python3
"""
This module contains the function high that
sorts a DataFrame by the High column in descending order.
"""


def high(df):
    """
    Sort the DataFrame by the High column in descending order.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with a 'High' column.

    Returns
    -------
    pandas.DataFrame
        DataFrame sorted by High in descending order.
    """
    return df.sort_values(by='High', ascending=False)
