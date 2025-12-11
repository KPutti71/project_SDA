#RQ4: Does trading volume affect stock returns?
# Hypothesis:
#     H0: Volume has no effect on returns (beta = 0)
#     H1: Volume affects returns (beta =! 0)

# Steps:
# 1) loads datasets ("stock_prices_raw.csv" and "stock_volumes_raw.csv")
# 2) cleans the data (drops NaNs)
# 4) matches price to volume for each stock (stock and stock.1)
# 5) computes log returns
# 6) runs the LRM 
# 7) get slope values
# 8) visualize and get plots 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
#imported lrm function from the lrm_code.py
from lrm_code import LRM_RSS          

# ----------------------------------------------------------
# 1. get and clean data

prices = pd.read_csv("datasets/stock_prices_raw.csv")
volumes = pd.read_csv("datasets/stock_volumes_raw.csv")

prices = prices.dropna()
volumes = volumes.dropna()

columns = prices.columns.tolist()

# ----------------------------------------------------------
# 2. match price and volums for each stock

price_columns = [
    c for c in columns
    if c != "Date" and not c.endswith(".1")
]

# all the volume columns have .1 (ex.stock.1)
volume_columns = [c for c in volumes.columns if c.endswith(".1")]

#matching the stocks by mapping together
pairs = {}
for ticker in price_columns:
    vol_col = ticker + ".1"
    if vol_col in volume_columns:
        pairs[ticker] = vol_col

print("Stock price and volume matched pairs:")
print(pairs)
print()

# ----------------------------------------------------------
# 3. get the regression
# we defined the regression LRM_RSS in lrm_code.py and imported it here

# ----------------------------------------------------------
# 4. log returns
# log returns are used because the original prices are not stationary.
for ticker in pairs.keys():
    prices[f"{ticker}_return"] = np.log(prices[ticker].astype(float)).diff()
df_returns = prices.dropna().copy()


# ----------------------------------------------------------
# 5. computing the regressions usinf LRM for every stock 

#define dict for results
results = []

for ticker, vol_col in pairs.items():
    Y = df_returns[f"{ticker}_return"].astype(float).values
    X_vals = volumes[vol_col].astype(float).values[-len(Y):]

    beta = LRM_RSS([X_vals], Y)
    slope = beta[1]

    results.append([ticker, round(slope, 10)])

results_df = pd.DataFrame(results, columns=["Ticker", "Slope (Volume → Return)"])

print("Q4: Does volume haeve an effect on daily log returns")
print(tabulate(results_df, headers="keys", tablefmt="psql"))
print()


# ----------------------------------------------------------
# 6. plot 

sample_stock = list(pairs.keys())[0]
sample_volume = pairs[sample_stock]
sample_return = f"{sample_stock}_return"

x = volumes[sample_volume].astype(float).values[-len(df_returns):]
y = df_returns[sample_return].astype(float).values

beta_ex = LRM_RSS([x], y)
line = beta_ex[0] + beta_ex[1] * x

plt.figure()
plt.scatter(x, y, s=10, alpha=0.5, label="daily observations")
plt.plot(x, line, color="red", linewidth=2, label="Fitted line")

plt.title(f"Q4: Volume vs Return for {sample_stock}")
plt.xlabel("Volume")
plt.ylabel("Log Return")
plt.legend()
plt.tight_layout()
plt.show()
