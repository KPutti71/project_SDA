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

# Remove row with indication of stock price or volume since we can identify with column names
df_transpose = df.T
df_transpose.pop(0)
df = df_transpose.T

# Get column names for later use
columns = df.columns

# Time (all series now have the same length after dropna)
time_index = np.arange(len(df))

 #%%
"""
Test plot for AMZN closing prices over time
"""

# Get closing prices for AMZN
AMZN_cls = df["AMZN"].astype('float')

# Plot a test figure for one example
plt.figure(figsize=(11, 5))
plt.plot(time_index, AMZN_cls)
plt.title('AMZN closing price over time')
plt.xlabel('Time index')
plt.ylabel('Closing price')
plt.show()

# %%
"""
Fitting LRM with a minimizing RSS method
"""

Y = AMZN_cls
x = [time_index]


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


beta = LRM_RSS(x, Y)

# Make plot of the fit
plt.figure(figsize=(11, 5))
plt.plot(time_index, AMZN_cls, label="Observed")
plt.plot(time_index, beta[0] + beta[1] * x[0], label="Fitted LRM")
plt.title('AMZN closing price over time with fitted LRM')
plt.xlabel('Time index')
plt.ylabel('Closing price')
plt.legend()
plt.show()

# %%
"""
Bootstrap over time series and make confidence interval for our test
"""


def bootstrap(Y, n=1000):
    """
    Does n bootstraps over set Y where each bootstrap has the same length as Y
    
    :param Y: Data set for bootstrap
    :param n: Amount of bootstraps
    """
    length = len(Y)
    all_boot_Y = []
    for i in range(n):
        all_boot_Y.append(np.random.choice(Y, size=length, replace=True))
    return all_boot_Y


def CI(set):
    """
    Gives the confidence interval of a set of test values
    
    :param set: A list of test values
    """
    return [np.percentile(set, 2.5), np.percentile(set, 97.5)]


# %%
"""
Bootstrap for LRM stationarity testing
"""


def stationarity_bootstrap(x, Y, n=1000):
    boot_sets = bootstrap(Y)
    boot_slopes = [LRM_RSS(x, boot)[1] for boot in boot_sets]
    interval = CI(boot_slopes)
    return boot_slopes, interval


boot_slopes, interval = stationarity_bootstrap(x, Y)

# Histogram of bootstrap slope distribution
plt.figure(figsize=(11, 5))
n_hist, bins, patches = plt.hist(boot_slopes, 40)
plt.vlines(interval[0], 0, max(n_hist), 'r')
plt.vlines(interval[1], 0, max(n_hist), 'r')
plt.title("Bootstrap distribution of slopes AMZN closing prices")
plt.xlabel("Slope")
plt.ylabel("Frequency")
plt.show()


# %%
"""
Check slopes of all prices and volumes over the time
"""


def reject_or_not(value, interval):
    if value < interval[0] or value > interval[1]:
        return 'Reject'
    else:
        return "Don't Reject"


all_slopes = []
all_intervals = []

# Skip first because this is the date column
for col in columns[1:]:
    data_Y = df[col].astype('float')
    data_x = [np.arange(len(data_Y))]
    
    # Compute slope 4-decimal precision
    slope = LRM_RSS(data_x, data_Y)[1]
    slope = round(slope, 4)
    
    # Bootstrap interval 4-decimal precision
    _, interval = stationarity_bootstrap(data_x, data_Y)
    interval = tuple(round(x.item(), 4) for x in interval)

    
    all_slopes.append(slope)
    all_intervals.append(interval)

reject_list = [reject_or_not(all_slopes[i], all_intervals[i]) for i in range(len(all_slopes))]

result_df = pd.DataFrame({
    "Ticker": columns[1:], 
    "Slope": all_slopes, 
    "Bootstrap Slope CI": all_intervals,
    "Reject or Not": reject_list
})

print(tabulate(result_df, headers = 'keys', tablefmt = 'psql'))
