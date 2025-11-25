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

bitcoin = pd.read_csv("project_SDA/datasets/Bitcoin.csv", sep=',')

bitcoin_open = bitcoin[['Date', 'Open']]

open_prices = bitcoin_open['Open']
time = np.arange(len(open_prices))

plt.plot(time, open_prices)
plt.show()

