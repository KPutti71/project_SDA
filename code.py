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

full_file = pd.read_csv("datasets/djia30_btc_gold.csv", sep=',')

# Remove the weekends since stock market is closed so we wont have any values
df = full_file.dropna()
print(df['AMZN'])

 #%%
"""
Test plot for bitcoin and gold open prices over time
"""

bitcoin_open = bitcoin['Open']
bitcoin_open_time = np.arange(len(bitcoin_open))

gold_open = gold['Open']
gold_open_time = np.arange(len(gold_open))

# plt.figure(figsize=(11,5))

# plt.subplot(1,2,1)
# plt.plot(bitcoin_open_time, bitcoin_open)
# plt.title('Bitcoin price over time')

# plt.subplot(1,2,2)
# plt.plot(gold_open_time, gold_open)
# plt.title('Gold price over time')

# plt.show()

# %%
"""
Fitting LRM for n features and m observations
Y = beta * Xf
"""

n = 1
m = len(gold_open_time)
x1 = gold_open_time

# C olumn one should be ones for the intercept
X = np.zeros((m,n+1))
X[:,0] = 1

# Loop over columns and rows to fill in obeserved data x
for col in range(n):
    for row in range(m):
        X[row, col+1] = x1[row]

# Calculate the parameter 
Y = gold_open
beta = np.linalg.inv(X.T @ X) @ X.T @ Y

# Make plot of the fit
plt.plot(gold_open_time, gold_open)
plt.plot(gold_open_time, beta[1] * gold_open_time + beta[0])
plt.title('Gold price over time')
plt.show()

# %%
