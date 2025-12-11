#RQ4: Does trading volume affect stock returns?
# Hypothesis:
#     H0: Volume has no effect on returns (beta = 0)
#     H1: Volume affects returns (beta =! 0)

# Steps:
# 1) loads dataset "djia30_btc_gold.csv"
# 2) cleans the data (drops NaNs + removes metadata row)
# 4) matches price to volume for each stock (stock and stock.1)
# 5) computes log returns
# 6) runs the LRM 
# 7) get slope values
# 8) visualize and get plots 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
# ----------------------------------------------------------
# 1. get and clean data

df = pd.read_csv("datasets/djia30_btc_gold.csv")

#drops weekends and missing values and metadata (rows/columns)
df = df.dropna()
df = df[df["Ticker"] != "Price"].copy()
columns = df.columns.tolist()

# ----------------------------------------------------------
# 2. match price and volums for each stock

price_columns = [
    c for c in columns
    if c != "Ticker" and not c.endswith(".1")
]

# all the volume columns have .1 (ex.stock.1)
volume_columns = [c for c in columns if c.endswith(".1")]

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
# 3.define the regression LRM_RSS (taken from our lrm_code.py file)

def LRM_RSS(x, Y):
    """
    Fit a Linear Regression Model (LRM) to the given observed data x and Y 
    using the RSS method: Y = beta * X in matrix notation.
    Returns the parameter beta as a vector of parameter values.
    
    :param x: nested list of feature arrays
    :param Y: array-like of y values
    """
    # n is the number of features and m is the number of observations
    # For all items in x we assume that they have the same dimension, which we ensured
    # when removing NaN values from df.
    n = len(x)
    m = len(x[0])

    # Design matrix X with a column of ones for the intercept
    X = np.zeros((m, n + 1))
    X[:, 0] = 1

    # Loop over columns to fill in observed data x
    for col in range(n):
        X[:, col + 1] = x[col]

    # Calculate the parameter vector beta using the closed-form OLS solution
    beta = np.linalg.inv(X.T @ X) @ (X.T @ Y)
    return beta


# ----------------------------------------------------------
# 4. log returns

for ticker in pairs.keys():
    df[f"{ticker}_return"] = np.log(df[ticker].astype(float)).diff()
df_returns = df.dropna().copy()


# ----------------------------------------------------------
# 5. computing the regressions usinf LRM for every stock 

#define dict for results
results = []

for ticker, vol_col in pairs.items():
    Y = df_returns[f"{ticker}_return"].astype(float).values
    X_vals = df_returns[vol_col].astype(float).values

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

x = df_returns[sample_volume].astype(float).values
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
