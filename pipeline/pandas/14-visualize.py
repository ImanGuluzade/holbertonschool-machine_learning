#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd

from_file = __import__('2-from_file').from_file

def visualize(df):
    """
    Transform and visualize cryptocurrency data from a DataFrame.
    Steps:
    - Remove Weighted_Price
    - Rename Timestamp -> Date
    - Convert timestamp to datetime
    - Set Date as index
    - Fill missing Close with previous row
    - Fill missing High, Low, Open with same row Close
    - Fill missing Volume columns with 0
    - Filter data from 2017 onward
    - Resample daily and aggregate
    """
    # Remove Weighted_Price
    if 'Weighted_Price' in df.columns:
        df = df.drop('Weighted_Price', axis=1)

    # Rename Timestamp -> Date
    df = df.rename(columns={'Timestamp': 'Date'})

    # Convert timestamp to datetime
    df['Date'] = pd.to_datetime(df['Date'], unit='s')

    # Set Date as index
    df = df.set_index('Date')

    # Fill missing Close with previous row
    df['Close'] = df['Close'].fillna(method='ffill')

    # Fill missing High, Low, Open with same row Close
    for col in ['High', 'Low', 'Open']:
        df[col] = df[col].fillna(df['Close'])

    # Fill missing Volume columns with 0
    for col in ['Volume_(BTC)', 'Volume_(Currency)']:
        df[col] = df[col].fillna(0)

    # Filter from 2017 onwards
    df = df[df.index >= '2017-01-01']

    # Resample daily and aggregate
    df_daily = df.resample('D').agg({
        'High': 'max',
        'Low': 'min',
        'Open': 'mean',
        'Close': 'mean',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum'
    })

    # Plot
    df_daily.plot(figsize=(12, 6))
    plt.title('Daily Aggregated Crypto Prices')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.tight_layout()
    plt.show()

    # Return transformed DataFrame
    return df_daily
