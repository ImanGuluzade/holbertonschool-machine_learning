#!/usr/bin/env python3
"""
This module contains the function flip_switch that
sorts a DataFrame in reverse chronological order
and transposes it.
"""


def flip_switch(df):
    """
    Sort the DataFrame in reverse chronological order
    and transpose it.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        Transformed DataFrame.
    """
    # Reverse the row order (latest first)
    df_sorted = df.iloc[::-1]

    # Transpose the DataFrame
    return df_sorted.T
