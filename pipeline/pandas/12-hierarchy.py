#!/usr/bin/env python3
"""
Concatenate two DataFrames with Timestamp as first level of MultiIndex
and select a specific timestamp range.
"""

import pandas as pd

# Import index function from exercise 10
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Concatenate df2 (bitstamp) and df1 (coinbase) for a given timestamp range
    with Timestamp as first level in the MultiIndex and chronological order.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Coinbase DataFrame.
    df2 : pandas.DataFrame
        Bitstamp DataFrame.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame with keys and Timestamp as first index level.
    """
    # Step 1: Set Timestamp as index for both
    df1_indexed = index(df1)
    df2_indexed = index(df2)

    # Step 2: Select the timestamp range
    start, end = 1417411980, 1417417980
    df1_range = df1_indexed.loc[start:end]
    df2_range = df2_indexed.loc[start:end]

    # Step 3: Concatenate with keys
    df_concat = pd.concat([df2_range, df1_range], keys=['bitstamp', 'coinbase'])

    # Step 4: Swap levels to have Timestamp first
    df_concat = df_concat.swaplevel(0, 1)

    # Step 5: Sort by Timestamp (chronological)
    df_concat = df_concat.sort_index()

    return df_concat
