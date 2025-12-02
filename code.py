"""
Imported packages for the code
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
"""
Read csv files
"""

bitcoin = pd.read_csv("datasets/bitcoin_prices.csv", sep=',')
gold = pd.read_csv("datasets/gold_prices.csv", sep=',')

# %%
"""
Reading new file containing all stocks/btc/gold prices and volumes
"""

full_file = pd.read_csv("datasets/djia30_btc_gold.csv", sep=',')

# Remove the rows with NAN values since stock market is closed so we wont have any values
df = full_file.dropna()

# Remove row with indication of stock price or volume since we can identify with column names
df_transpose = df.T
df_transpose.pop(0)
df = df_transpose.T

 #%%
"""
Test plot for AMZN closing prices over time
"""

# Get closing prices for AMZN and calculate the evaluated time
AMZN_cls = df["AMZN"][1:].astype('float')
AMZN_cls_time = np.arange(len(AMZN_cls))

# Plot a test figure for one example
plt.figure(figsize=(11,5))
plt.plot(AMZN_cls_time, AMZN_cls)
plt.title('AMZN closing price over time')
plt.show()

# %%
"""
Fitting LRM with a minimizing RSS method
Y = beta * X
"""

# n is the number of features and m is the dimension of the given features
# for all items in x we assume that they have the same dimension which me made sure of in the removing NAN values in df
x = [AMZN_cls_time]
n = len(x)
m = len(x[0])
Y = AMZN_cls

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
print(beta)

# Make plot of the fit
plt.figure(figsize=(11,5))
plt.plot(AMZN_cls_time, AMZN_cls)
plt.plot(AMZN_cls_time,  + beta[0] + beta[1] * x[0])
plt.title('AMZN closing price over time with fitted LRM')
plt.show()

# %%
