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

 #%%
"""
Test plot for bitcoin and gold open prices over time
"""

bitcoin_open = bitcoin['Open']
bitcoin_open_time = np.arange(len(bitcoin_open))

gold_open = gold['Open']
gold_open_time = np.arange(len(gold_open))

plt.figure(figsize=(11,5))

plt.subplot(1,2,1)
plt.plot(bitcoin_open_time, bitcoin_open)
plt.title('Bitcoin price over time')

plt.subplot(1,2,2)
plt.plot(gold_open_time, gold_open)
plt.title('Gold price over time')

plt.show()

# %%
"""

"""
