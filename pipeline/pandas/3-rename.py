#!/usr/bin/env python3
"""
This module contains the function rename that
renames the Timestamp column to Datetime, converts
timestamps to datetime, and returns only Datetime and Close.
"""

import pandas as pd


def rename(df):
    """
    Rename 'Timestamp' to 'Datetime', convert to datetime,
    and keep only 'Datetime' and 'Close' columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a 'Timestamp' column.

    Returns
    -------
    pandas.DataFrame
        Modified DataFrame with 'Datetime' and 'Close' columns.
    """
    # Rename column
    df = df.rename(columns={"Timestamp": "Datetime"})
    
    # Convert to datetime
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit='s')
    
    # Keep only Datetime and Close
    df = df[["Datetime", "Close"]]
    
    return df
