"""
Dataset Preparation: Stock Prices and Volumes

This module:
1. Reads the combined DJIA / Bitcoin / Gold dataset
2. Removes rows with missing values
3. Separates price and volume data
4. Exports cleaned datasets for downstream analysis

Datasets used:
- datasets/djia30_btc_gold.csv

Outputs:
- datasets/stock_prices_raw.csv
- datasets/stock_volumes_raw.csv
"""

# ================= Imports =================

import pandas as pd


# ================= Data Loading =================

full_file = pd.read_csv("datasets/djia30_btc_gold.csv", sep=",")

# Remove rows with NaN values (weekends / market closures)
df = full_file.dropna()

# Get column names for later use
columns = list(df.columns)


# ================= Data Processing =================

# First 36 columns contain prices
df_prices = df[columns[:36]]

# Remaining columns contain volumes (keep date column)
df_volumes = df[[columns[0]] + columns[36:]]


# ================= Data Export =================

df_prices.to_csv("datasets/stock_prices_raw.csv", index=False)
df_volumes.to_csv("datasets/stock_volumes_raw.csv", index=False)


# ================= Main Execution =================

def main():
    print("Columns in cleaned dataset:")
    print(columns)
    print("Price and volume datasets successfully saved.")


if __name__ == "__main__":
    main()
