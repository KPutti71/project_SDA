"""
Autoregressive Modeling and Forecasting via Likelihood Optimization

This module:
1. Fits a linear regression model via likelihood maximization
2. Applies the model to a lag-1 autoregressive process
3. Simulates future paths with parameter uncertainty
4. Produces forecasts and confidence intervals

Dataset used:
- datasets/stock_prices_raw.csv
"""

# ================= Imports =================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ================= Data Loading =================

df = pd.read_csv("datasets/stock_prices_raw.csv", sep=',')

columns = df.columns
time_index = np.arange(len(df))


# ================= Linear Regression Likelihood =================

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


# ================= Autoregression =================

def lag_1(data):
    X_t   = np.array(data[1:])
    X_t_1 = np.array(data[:-1])
    return np.column_stack([X_t_1, X_t])


# ================= Forecasting =================

def single_path_forecast(alpha, beta, sigma, last_value, steps):
    """Generate a single forecast path."""
    path = [last_value]
    for _ in range(steps - 1):
        noise = np.random.normal(0, sigma)
        next_val = alpha + beta * path[-1] + noise
        path.append(next_val)
    return np.array(path)


def multi_path_forecast(data, lagged_sample, steps=30, num_param_sets=25, iterations=1000):
    """
    Generate multiple forecast paths with parameter uncertainty.
    """
    param_list = [fit_lrm(lagged_sample) for _ in range(num_param_sets)]

    all_paths = np.zeros((num_param_sets, iterations, steps))

    for k, (alpha, beta, sigma) in enumerate(param_list):
        for j in range(iterations):
            all_paths[k, j] = single_path_forecast(
                alpha, beta, sigma, data[-1], steps
            )

    return all_paths.reshape(-1, steps)


# ================= Plotting =================

def plot_lag_fit(sample, alpha, beta):
    """
    Plot lag-1 scatter and fitted regression line.
    """
    plt.scatter(sample[:, 0], sample[:, 1])
    plt.plot(sample[:, 0], alpha + beta * sample[:, 0], "r")
    plt.title("Lag-1 Autoregression Fit")
    plt.xlabel(r"$X_{t-1}$")
    plt.ylabel(r"$X_t$")
    plt.show()


def plot_forecast_paths(time, data, forecast_paths):
    """
    Plot forecast paths with confidence intervals.
    """
    steps = forecast_paths.shape[1]
    new_time = np.arange(steps) + len(data) - 1

    mean_path = forecast_paths.mean(axis=0)
    ci_low = np.percentile(forecast_paths, 2.5, axis=0)
    ci_high = np.percentile(forecast_paths, 97.5, axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(time[-150:], data[-150:], label="Observed")

    for path in forecast_paths:
        plt.plot(new_time, path, color=(0, 0, 1, 0.02), linewidth=1)

    plt.plot(new_time, mean_path, "y", label="Mean forecast")
    plt.plot(new_time, ci_low, "r", label="95% CI")
    plt.plot(new_time, ci_high, "r")

    plt.title("Forecast Paths with Parameter Uncertainty")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


# ================= Main Execution =================

def main():
    data = list(df["AAPL"].astype(float))
    time = np.arange(len(data))

    lagged = lag_1(data)
    alpha, beta, sigma = fit_lrm(lagged)

    plot_lag_fit(lagged, alpha, beta)

    forecast_paths = multi_path_forecast(data, lagged, steps=30, num_param_sets=25, iterations=1000)

    plot_forecast_paths(time, data, forecast_paths)


if __name__ == "__main__":
    main()