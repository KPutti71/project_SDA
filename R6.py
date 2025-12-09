"""
R6: Does trading volume affect stock volatility?

This script:
1. Loads daily log price data and raw volume data for multiple tickers.
2. Aligns tickers and dates across the two datasets.
3. Computes daily log returns from log prices and constructs a 30-day rolling, annualised volatility measure for each ticker.
4. Computes a 30-day rolling average of trading volume and takes its natural logarithm to reduce skew.
5. Reshapes the data into a panel (Date, ticker, volatility, log volume).
6. Implements Ordinary Least Squares (OLS) regression from scratch using NumPy to estimate the pooled linear model:
       vol_30d = alpha + beta * log_vol_avg_30d + error
    where:
        vol_30d          = 30-day rolling annualised volatility
        log_vol_avg_30d  = log 30-day average volume
    The key hypothesis for R6 is:
        H0: beta = 0  (trading volume has no effect on volatility)
        H1: beta ≠ 0  (trading volume affects volatility)

    The script computes coefficient estimates, standard errors, t-statistics, and R-squared for this model.
7. Produces a scatter plot of volatility against log volume for a random subsample of observations, with the fitted OLS regression line overlaid, to visualise the volume-volatility relationship.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def ols(y: np.ndarray, X: np.ndarray) -> dict:
    """
    Basic OLS implementation:
        y: (n,) array
        X: (n, k) array (first column should be ones for intercept)
    Returns beta, standard errors, t-stats, R^2, n, k.
    """
    y = y.reshape(-1, 1)  # (n, 1)
    X = np.asarray(X)  # (n, k)
    n, k = X.shape

    # Beta = (X'X)^(-1) X'y
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    Xty = X.T @ y
    beta = XtX_inv @ Xty  # (k, 1)

    # Fitted values and residuals
    y_hat = X @ beta  # (n, 1)
    resid = y - y_hat  # (n, 1)

    # Residual variance
    rss = float(resid.T @ resid)
    sigma2 = rss / (n - k)

    # Var(beta) = sigma^2 (X'X)^(-1)
    var_beta = sigma2 * XtX_inv
    se_beta = np.sqrt(np.diag(var_beta)).reshape(-1, 1)

    # t-statistics
    t_stats = beta / se_beta

    # R-squared
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss

    return {
        "beta": beta.flatten(),  # [alpha, beta]
        "se": se_beta.flatten(),
        "t": t_stats.flatten(),
        "r2": r2,
        "n": n,
        "k": k,
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

    # Design matrix with intercept
    X = np.column_stack([np.ones_like(x), np.asarray(x)])

    results = ols(y, X)

    alpha_hat, beta_hat = results["beta"]
    se_alpha, se_beta = results["se"]
    t_alpha, t_beta = results["t"]

    print("\nOLS results (pooled):")
    print(f"alpha = {alpha_hat:.6f} (SE = {se_alpha:.6f}, t = {t_alpha:.2f})")
    print(f"beta  = {beta_hat:.6f} (SE = {se_beta:.6f}, t = {t_beta:.2f})")
    print(f"R^2   = {results['r2']:.3f}")
    print(f"n = {results['n']}, k = {results['k']}")

    plot_relationship(panel, alpha_hat, beta_hat)


if __name__ == "__main__":
    main()
