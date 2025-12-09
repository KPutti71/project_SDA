"""
Imported packages for the code
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# %%
"""
Reading new file containing all stocks/BTC/gold prices and volumes
"""

full_file = pd.read_csv("datasets/djia30_btc_gold.csv", sep=',')

# Remove the rows with NaN values since stock market is closed in weekend so we won't have all values
df = full_file.dropna()

# Get column names for later use
columns = list(df.columns)
print(columns)

# Get columns of all the prices
df_prices = df[columns[:36]]
df_volumes = df[[columns[0]] + columns[36:]]

df_prices.to_csv('datasets/stock_prices_raw.csv', index=False)
df_volumes.to_csv('datasets/stock_volumes_raw.csv', index=False)