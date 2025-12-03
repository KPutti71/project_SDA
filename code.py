"""
Imported packages for the code
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
"""
Reading new file containing all stocks/btc/gold prices and volumes
"""

full_file = pd.read_csv("datasets/djia30_btc_gold.csv", sep=',')

# Remove the rows with NAN values since stock market is closed in weekend so we wont have all values
df = full_file.dropna()

# Remove row with indication of stock price or volume since we can identify with column names
df_transpose = df.T
df_transpose.pop(0)
df = df_transpose.T

# Get column names for later use
columns = df.columns

 #%%
"""
Test plot for AMZN closing prices over time
"""

# Get closing prices for AMZN and calculate the evaluated time
AMZN_cls = df["AMZN"].astype('float')
AMZN_cls_time = np.arange(len(AMZN_cls))

# Plot a test figure for one example
plt.figure(figsize=(11,5))
plt.plot(AMZN_cls_time, AMZN_cls)
plt.title('AMZN closing price over time')
plt.show()

# %%
"""
Fitting LRM with a minimizing RSS method
"""

Y = AMZN_cls
x = [AMZN_cls_time]


def LRM_RSS(x, Y):
    """
    Fit a LRM to the given observed data x and Y using the RSS method: Y = beta * X in matrix notation,
    returns the parameter beta as a vector of parameter values.
    
    :param x: nested list of features
    :param Y: list y values
    """
    # n is the number of features and m is the dimension of the given features
    # for all items in x we assume that they have the same dimension which me made sure of in the removing NAN values in df
    n = len(x)
    m = len(x[0])

    # Column one should be ones for the intercept
    X = np.zeros((m,n+1))
    X[:,0] = 1
    print(X)

    # Loop over columns to fill in obeserved data x
    for col in range(n):
        X[:, col+1] = x[col]
    print(X)

    # Calculate the parameter 
    beta = np.linalg.inv(X.T @ X) @ X.T @ Y
    return beta


beta = LRM_RSS(x, Y)
print(beta[1])

# Make plot of the fit
plt.figure(figsize=(11,5))
plt.plot(AMZN_cls_time, AMZN_cls)
plt.plot(AMZN_cls_time, beta[0] + beta[1] * x[0])
plt.title('AMZN closing price over time with fitted LRM')
plt.show()

# %%
"""
Check slopes of all prices and volumes over the time
"""

all_slopes = []
# Skip first because this is the date column
for col in columns[1:]:
    # Get data of all columns
    data_Y = df[col].astype('float')
    data_x = [np.arange(len(data_Y))]
    param = LRM_RSS(data_x, data_Y)
    slope = param[1]
    all_slopes.append(slope)

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


def bootstrap_stationarity(x, Y, n=1000):
    boot_sets = bootstrap(Y)
    boot_slopes = [LRM_RSS(x, boot)[1] for boot in boot_sets]
    interval = CI(boot_slopes)
    return boot_slopes, interval


Y = AMZN_cls
x = [AMZN_cls_time]

boot_slopes, interval = bootstrap_stationarity(x, Y)
print(interval)

n, bins, patches = plt.hist(boot_slopes, 40)
plt.vlines(interval[0], 0, max(n), 'r')
plt.vlines(interval[1], 0, max(n), 'r')
plt.show()
