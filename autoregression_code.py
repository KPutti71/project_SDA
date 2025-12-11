"""
Imported packages for the code
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
"""
Reading new file containing all stocks/BTC/gold prices and volumes
"""

df = pd.read_csv("datasets/stock_prices_raw.csv", sep=',')

# Get column names for later use
columns = df.columns

# Time (all series now have the same length after dropna)
time_index = np.arange(len(df))

# %%

def loglikelihood_lrm(alpha, beta, sigma_y, sample):
    """
    Calculates the loglikelihood for a linear regression model
    
    :param alpha: Model parameter
    :param beta: Model parameter
    :param sigma_y: Model parameter
    :param sample: List of coupled data points
    """
    x = sample[:,0]
    y = sample[:,1]
    mu = alpha + beta * x

    n = len(x)
    return -n * np.log(np.sqrt(2*np.pi)*sigma_y) - np.sum((y - mu)**2) / (2*sigma_y**2)


def fit_lrm(sample):
    """
    Fit a linear regression model to a given set of coupled datapoints, the method that is used for this fitting
    starts of with 3 random parameters between 0 and 5 and then does a small stap and checks if this improves our 
    loglikelihood. If it improves take new point as starting point.
    
    :param sample: List of coupled data points
    """

    # initial guess
    alpha = np.random.rand() * 5
    beta  = np.random.rand() * 5
    sigma_y = np.random.rand() * 5

    tries = 100      # tries per iteration
    N = 100          # iterations

    for n in range(N):

        for t in range(tries):

            # evaluate log-likelihood
            log_L = loglikelihood_lrm(alpha, beta, sigma_y, sample)

            # induce a small random step
            delta = np.random.uniform(-0.5, 0.5, 3)

            new_alpha = alpha + delta[0]
            new_beta  = beta  + delta[1]
            new_sigma = sigma_y + delta[2]

            # make sure sigma > 0
            if new_sigma <= 0:
                continue

            # evaluate new log-likelihood
            new_log_L = loglikelihood_lrm(new_alpha, new_beta, new_sigma, sample)

            # accept if improves likelihood
            if new_log_L > log_L:
                alpha, beta, sigma_y = new_alpha, new_beta, new_sigma
                break

    return alpha, beta, sigma_y

#%%
"""
Autoregression
"""

data = list(df['AAPL'].astype('float'))
time = np.arange(len(data))


def lag_1(data):
    X_t   = np.array(data[1:])
    X_t_1 = np.array(data[:-1])
    return np.column_stack([X_t_1, X_t])


lag1 = lag_1(data)
alpha_test, beta_test, sigma_y = fit_lrm(lag1)
print(alpha_test, beta_test, sigma_y)

plt.scatter(lag1[:,0], lag1[:,1])
plt.plot(lag1[:,0], alpha_test + beta_test * lag1[:,0], 'r')
plt.show()

# %%
"""
Makes a plot of a prediction of the next 30 points in our time series
"""

next_steps = 30
new_time = np.arange(next_steps) + len(time) - 1

new_points = [data[-1]]
for t in new_time[1:]:
    noise = np.random.normal(0, sigma_y)
    point = alpha_test + beta_test * new_points[-1] + 0.1 * noise
    new_points.append(point)

plt.plot(time[-90:], data[-90:])

plt.plot(new_time, new_points, 'r')
plt.show()

# %%
"""
Makes a plot of 25 times 1000 prediction of the next 30 points in our time series where echt 1000 predictions uses
a new parameter set
"""

num_param_sets = 25
param_list = []

for _ in range(num_param_sets):
    alpha_i, beta_i, sigma_i = fit_lrm(lag1)
    param_list.append((alpha_i, beta_i, sigma_i))

# prediction setup
next_steps = 30
iterations = 1000
all_paths = np.zeros((num_param_sets, iterations, next_steps))

# 25 × 1000 prediction paths
for k, (alpha_i, beta_i, sigma_i) in enumerate(param_list):
    for j in range(iterations):
        new_points = np.zeros(next_steps)
        
        # use last value
        new_points[0] = alpha_i + beta_i * data[-1] + np.random.normal(0, sigma_i)

        for t in range(1, next_steps):
            noise = np.random.normal(0, sigma_i)
            new_points[t] = alpha_i + beta_i * new_points[t-1] + noise
        
        all_paths[k, j] = new_points

# combine all predictions ---
all_pred = all_paths.reshape(-1, next_steps)

# compute CI
mean_path = all_pred.mean(axis=0)
ci_low  = np.percentile(all_pred, 2.5, axis=0)
ci_high = np.percentile(all_pred, 97.5, axis=0)

# new_time setup
new_time = np.arange(next_steps) + len(data) - 1

# Plot
plt.figure(figsize=(8,5))

plt.plot(time[-150:], data[-150:])

for p in all_paths.reshape(-1, all_paths.shape[-1]):   # flatten to 25000 paths
    plt.plot(new_time, p, '-', color=(0,0,1,0.02), linewidth=1)

plt.plot(new_time, mean_path, 'y', label='Mean')
plt.plot(new_time, ci_low,  'r', label='Confidence Interval')
plt.plot(new_time, ci_high, 'r')

plt.title("All 25 000 Prediction Paths")
plt.xlabel("Time")
plt.ylabel("Predicted Value")
plt.legend()

plt.show()