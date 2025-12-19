"""
R6: Does trading volume affect stock volatility?

This script:
1. Loads daily log price data and raw volume data for multiple tickers.
2. Aligns tickers and dates across the two datasets.
3. Computes daily log returns from log prices and constructs a 30-day rolling, annualised volatility measure for each ticker.
4. Computes a 30-day rolling average of trading volume and takes its natural logarithm to reduce skew.
5. Reshapes the data into a panel (Date, ticker, volatility, log volume).
6. Implements a linear regression model using the autoregression fitting approach (from autoregression_code.py)
   to estimate the pooled linear model:
       vol_30d = alpha + beta * log_vol_avg_30d + error
    where:
        vol_30d          = 30-day rolling annualised volatility
        log_vol_avg_30d  = log 30-day average volume

    The fitting approach uses a maximum likelihood estimation with random walk optimization.

    The key hypothesis for R6 is:
        H0: beta = 0  (trading volume has no effect on volatility)
        H1: beta ≠ 0  (trading volume affects volatility)

    The script computes coefficient estimates, standard errors, t-statistics, and R-squared for this model.
7. Produces a scatter plot of volatility against log volume for a random subsample of observations, with the fitted regression line overlaid, to visualise the volume-volatility relationship.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import sys
import os
from unittest.mock import patch

with patch('matplotlib.pyplot.show'), open(os.devnull, 'w') as devnull:
    original_stdout = sys.stdout
    try:
        sys.stdout = devnull
        import autoregression_code as ar_code
        import lrm_code
    except Exception as e:
        sys.stdout = original_stdout
        print(f"Warning: Failed to import shared modules: {e}")
        raise e
    finally:
        sys.stdout = original_stdout


def load_data(
    price_path: str, volume_path: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load price (log) and volume data, align tickers, and return numeric DataFrames."""
    prices = pd.read_csv(price_path, parse_dates=["Date"])
    volumes = pd.read_csv(volume_path, parse_dates=["Date"])

    # Standardise volume column names by stripping ".1"
    rename_cols = {c: c.replace(".1", "") for c in volumes.columns if c != "Date"}
    volumes = volumes.rename(columns=rename_cols)

    # Common tickers across both files
    price_tickers = set(prices.columns) - {"Date"}
    volume_tickers = set(volumes.columns) - {"Date"}
    tickers = sorted(price_tickers & volume_tickers)

    # Use only common tickers, indexed by date
    prices = prices.set_index("Date").sort_index()[tickers]
    volumes = volumes.set_index("Date").sort_index()[tickers]

    # Ensure numeric
    prices = prices.apply(pd.to_numeric, errors="coerce")
    volumes = volumes.apply(pd.to_numeric, errors="coerce")

    return prices, volumes, tickers


def build_panel(
    prices: pd.DataFrame, volumes: pd.DataFrame, window: int = 30
) -> pd.DataFrame:
    """
    Construct a panel with:
        - 30-day rolling annualised volatility from log returns
        - log 30-day average volume
    """
    # prices are log prices; daily log returns are first differences
    log_ret = prices.diff()

    # 30-day rolling volatility (annualised)
    vol_30d = log_ret.rolling(window=window, min_periods=window).std() * np.sqrt(252)

    # 30-day rolling mean volume and its log
    vol_avg_30d = volumes.rolling(window=window, min_periods=window).mean()
    log_vol_avg_30d = np.log(vol_avg_30d.replace(0, np.nan))

    # Long format (Date, ticker, vol_30d, log_vol_avg_30d)
    vol_30d_long = vol_30d.stack().rename("vol_30d")
    log_vol_long = log_vol_avg_30d.stack().rename("log_vol_avg_30d")

    panel = pd.concat([vol_30d_long, log_vol_long], axis=1)
    panel = panel.dropna().reset_index()
    panel = panel.rename(columns={"level_0": "Date", "level_1": "ticker"})

    return panel


def fit_regression_model(y: np.ndarray, x: np.ndarray) -> dict:
    """
    Fit regression model using the autoregression fitting approach from autoregression_code.py
        y: (n,) array - dependent variable
        x: (n,) array - independent variable
    Returns alpha, beta, sigma_y, and computes R^2, n.
    """
    # Create sample in the format expected by fit_lrm: column_stack([x, y])
    sample = np.column_stack([x, y])

    # Fit the model using the autoregression approach
    alpha, beta, sigma_y = ar_code.fit_lrm(sample)

    # Compute fitted values and residuals
    y_hat = alpha + beta * x
    resid = y - y_hat

    # R-squared
    rss = np.sum(resid**2)
    tss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - rss / tss

    # Standard errors (approximate using sigma_y)
    n = len(y)
    x_mean = x.mean()
    sxx = np.sum((x - x_mean) ** 2)

    # SE for beta
    se_beta = sigma_y / np.sqrt(sxx)

    # SE for alpha
    se_alpha = sigma_y * np.sqrt(1 / n + x_mean**2 / sxx)

    # t-statistics
    t_alpha = alpha / se_alpha
    t_beta = beta / se_beta

    return {
        "alpha": alpha,
        "beta": beta,
        "sigma": sigma_y,
        "se_alpha": se_alpha,
        "se_beta": se_beta,
        "t_alpha": t_alpha,
        "t_beta": t_beta,
        "r2": r2,
        "n": n,
    }


def plot_relationship(
    panel: pd.DataFrame, alpha_hat: float, beta_hat: float, sample_size: int = 5000
) -> None:
    """Scatter plot of volatility vs log volume with fitted OLS line."""
    sample = panel.sample(min(sample_size, len(panel)), random_state=0)

    x = sample["log_vol_avg_30d"].to_numpy()
    y = sample["vol_30d"].to_numpy()

    x_min, x_max = x.min(), x.max()
    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = alpha_hat + beta_hat * x_grid

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.3, s=10, label="Observations")
    plt.plot(x_grid, y_grid, linewidth=2, label="Fitted line")

    plt.xlabel("Log 30-day average volume")
    plt.ylabel("30-day volatility (annualised)")
    plt.title("Volume–volatility relationship (OLS)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def print_beta_interpretation(beta: float) -> None:
    """
    Interpret beta in the semi-log model:
        vol_30d = alpha + beta * log_vol_avg_30d + error

    For a p% increase in volume:
        delta_logV = ln(1 + p/100)
        delta_vol  = beta * delta_logV
    For a kx increase in volume:
        delta_logV = ln(k)
        delta_vol  = beta * ln(k)
    """
    pct_changes = [1, 10, 50]
    multipliers = [2]

    print("\nInterpretation of beta (semi-log effect sizes):")
    for p in pct_changes:
        dlogv = np.log(1.0 + p / 100.0)
        dvol = beta * dlogv
        print(f"If 30-day avg volume is {p}% higher, predicted vol_30d increases by {dvol:.6f}.")

    for k in multipliers:
        dlogv = np.log(float(k))
        dvol = beta * dlogv
        print(f"If 30-day avg volume is multiplied by {k}x, predicted vol_30d increases by {dvol:.6f}.")

def main() -> None:
    prices, volumes, tickers = load_data(
        "datasets/stock_prices_log.csv",
        "datasets/stock_volumes_raw.csv",
    )

    print("Tickers used:", tickers)

    panel = build_panel(prices, volumes, window=30)
    print("Number of observations in panel:", len(panel))
    print(panel.head())

    # Dependent and independent variables
    y = np.asarray(panel["vol_30d"].values)
    x = np.asarray(panel["log_vol_avg_30d"].values)

    # Fit regression using autoregression approach
    results = fit_regression_model(y, x)

    alpha_hat = results["alpha"]
    beta_hat = results["beta"]
    se_alpha = results["se_alpha"]
    se_beta = results["se_beta"]
    t_alpha = results["t_alpha"]
    t_beta = results["t_beta"]

    print("\nRegression results (using autoregression model):")
    print(f"alpha = {alpha_hat:.6f} (SE = {se_alpha:.6f}, t = {t_alpha:.2f})")
    print(f"beta  = {beta_hat:.6f} (SE = {se_beta:.6f}, t = {t_beta:.2f})")
    print(f"sigma = {results['sigma']:.6f}")
    print(f"R^2   = {results['r2']:.3f}")
    print(f"n = {results['n']}")

    print_beta_interpretation(beta_hat)
    plot_relationship(panel, alpha_hat, beta_hat)


if __name__ == "__main__":
    main()
