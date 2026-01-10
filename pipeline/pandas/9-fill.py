#!/usr/bin/env python3
"""
This module contains the function fill that
removes Weighted_Price and fills missing values in a DataFrame.
"""


def fill(df):
    """
    Fill missing values in a DataFrame according to the following rules:
    - Remove the Weighted_Price column
    - Fill Close NaNs with the previous row's value
    - Fill High, Low, Open NaNs with the corresponding Close value in the
      same row
    - Set missing Volume_(BTC) and Volume_(Currency) to 0

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing trading data.

    Returns
    -------
    pandas.DataFrame
        DataFrame with missing values filled.
    """
    # Remove Weighted_Price column if it exists
    if 'Weighted_Price' in df.columns:
        df = df.drop(columns=['Weighted_Price'])

    # Fill Close missing values with previous row
    df['Close'] = df['Close'].fillna(method='ffill')

    # Fill High, Low, Open missing values with Close value in the same row
    for col in ['High', 'Low', 'Open']:
        df[col] = df[col].fillna(df['Close'])

    # Fill missing volumes with 0
    for col in ['Volume_(BTC)', 'Volume_(Currency)']:
        df[col] = df[col].fillna(0)

    return df
