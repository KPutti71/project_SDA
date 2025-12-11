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

df = pd.read_csv("datasets/stock_prices_raw.csv", sep=',')

# Get column names and time index for later use
columns = df.columns
time_index = np.arange(len(df))

# %%
"""
Fitting LRM with a minimizing RSS method
"""


def LRM_RSS(x, Y):
    """
    Fit a Linear Regression Model (LRM) to the given observed data x and Y 
    using the RSS method: Y = beta * X in matrix notation.
    Returns the parameter beta as a vector of parameter values.
    
    :param x: list of features (which are also lists)
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


# %%
"""
Sliding window method
"""

Y = df["MMM"].astype('float')
x = [time_index]


def rolling_mean(Y, size=100):
    return Y.rolling(window=size).mean()


def test_mean_stationarity(Y, window=30, boot_size=5000):
    window_means = rolling_mean(Y, window).dropna()
    x = [np.arange(len(window_means))]
    beta = LRM_RSS(x, window_means)[1]

    # Bootstrap
    b_betas = []
    for _ in range(boot_size):
        bootstrap = pd.Series(np.random.choice(Y, len(Y), replace=True))
        b_window_means = rolling_mean(bootstrap, window).dropna()
        b_x = [np.arange(len(b_window_means))]
        b_beta = LRM_RSS(b_x, b_window_means)[1]
        b_betas.append(b_beta)
    
    CI = [np.percentile(b_betas, 2.5), np.percentile(b_betas, 97.5)]
    result = "Reject H0 (mean not stationary)" \
        if beta < CI[0] or beta > CI[1] \
        else "Do not reject H0 (mean likely stationary)"

    return beta, b_betas, CI, result


def result_matrix():
    all_slopes = []
    all_intervals = []
    reject_list = []

    # Skip first column because it's the date column
    for col in columns[1:]:
        data_Y = df[col].astype(float)

        beta, _, CI, result = test_mean_stationarity(data_Y, 300)

        CI = tuple(round(x.item(), 4) for x in CI)

        all_slopes.append(beta)
        all_intervals.append(CI)
        reject_list.append(result)

    result_df = pd.DataFrame({
        "Ticker": columns[1:], 
        "Slope": all_slopes, 
        "Bootstrap Slope CI": all_intervals,
        "Reject or Not": reject_list
    })

    print(tabulate(result_df, headers="keys", tablefmt="psql"))


# result_matrix()

# %%


def make_plots():

    beta = LRM_RSS(x, Y)

    # Make plot of the fit
    plt.figure(figsize=(11, 5))
    plt.plot(time_index, Y, label="Observed")
    plt.plot(time_index, beta[0] + beta[1] * x[0], label="Fitted LRM")
    plt.title('MMM closing price over time with fitted LRM')
    plt.xlabel('Time index')
    plt.ylabel('Closing price')
    plt.legend()
    plt.show()

    _, boot_slopes, interval, _ = test_mean_stationarity(Y)

    plt.figure(figsize=(11, 5))
    n_hist, bins, patches = plt.hist(boot_slopes, 300)
    plt.vlines(interval[0], 0, max(n_hist), 'r')
    plt.vlines(interval[1], 0, max(n_hist), 'r')
    plt.title("Bootstrap distribution of sliding mean slopes MMM closing prices")
    plt.xlabel("Slope")
    plt.ylabel("Frequency")
    plt.show()

# make_plots()
