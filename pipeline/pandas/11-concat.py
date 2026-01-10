#!/usr/bin/env python3
"""
Concatenate two DataFrames with Timestamp index and keys.
"""

import pandas as pd

# Import the index function from exercise 10
index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenate df2 (bitstamp) on top of df1 (coinbase) using Timestamp index.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Coinbase DataFrame.
    df2 : pandas.DataFrame
        Bitstamp DataFrame.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame with keys 'bitstamp' and 'coinbase'.
    """
    # Set Timestamp as index
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    # Select rows from df2 up to timestamp 1417411920
    df2_selected = df2_indexed.loc[:1417411920]

    # Concatenate df2 on top of df1 with keys
    df_concat = pd.concat([df2_selected, df1_indexed], keys=['bitstamp', 'coinbase'])

    return df_concat
