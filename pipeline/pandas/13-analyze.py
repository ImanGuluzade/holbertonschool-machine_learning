#!/usr/bin/env python3
"""
Module that contains the analyze function for a DataFrame.
Computes descriptive statistics for all columns except Timestamp.
"""

def analyze(df):
    """
    Compute descriptive statistics for a DataFrame excluding the Timestamp column.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to analyze.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing descriptive statistics.
    """
    # Drop the Timestamp column if it exists
    df_numeric = df.drop(columns='Timestamp', errors='ignore')
    # Compute descriptive statistics using the DataFrame's describe method
    return df_numeric.describe()
