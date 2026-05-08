"""Technical indicator computation: MACD and Stochastic."""

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """
    Compute MACD(fast, slow, signal).
    Returns DataFrame with columns: macd, macd_signal, macd_hist.
    """
    close = df["close"]
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    result = df.copy()
    result["macd"] = macd_line
    result["macd_signal"] = signal_line
    result["macd_hist"] = histogram
    return result


def compute_stochastic(df: pd.DataFrame, k_period=14, d_period=3, d_slow_period=5) -> pd.DataFrame:
    """
    Compute Stochastic(k_period, d_period, d_slow_period).
    %K = (Close - Low_n) / (High_n - Low_n) * 100
    %D = SMA(d_period) of %K
    %D_slow = SMA(d_slow_period) of %D
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    low_n = low.rolling(window=k_period).min()
    high_n = high.rolling(window=k_period).max()
    denom = high_n - low_n
    denom = denom.replace(0, np.nan)

    pct_k = ((close - low_n) / denom) * 100
    pct_d = pct_k.rolling(window=d_period).mean()
    pct_d_slow = pct_d.rolling(window=d_slow_period).mean()

    result = df.copy()
    result["stoch_k"] = pct_k
    result["stoch_d"] = pct_d
    result["stoch_d_slow"] = pct_d_slow
    return result


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute both MACD and Stochastic on a DataFrame."""
    df = compute_macd(df)
    df = compute_stochastic(df)
    return df


def get_indicator_series(df: pd.DataFrame) -> dict:
    """
    Return dict of {indicator_name: np.array} after dropping NaNs.
    Only returns rows where ALL indicators are valid.
    """
    df = compute_all_indicators(df)
    indicator_cols = ["macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d", "stoch_d_slow"]
    df_clean = df[indicator_cols].dropna()
    return {col: df_clean[col].values for col in indicator_cols}


INDICATOR_TARGETS = {
    "macd": "macd",
    "stochastic": "stoch_k",
}
