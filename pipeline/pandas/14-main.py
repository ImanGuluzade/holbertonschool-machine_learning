#!/usr/bin/env python3
from_file = __import__('2-from_file').from_file
visualize = __import__('14-visualize').visualize

# Load CSV
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Transform and plot
df_daily = visualize(df)

# Print transformed DataFrame
print(df_daily)
