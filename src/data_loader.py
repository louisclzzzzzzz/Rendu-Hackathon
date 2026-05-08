"""Data loading, period filtering, weekly resampling, and preprocessing."""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# ── Market periods ─────────────────────────────────────────────────────────────
PERIODS = {
    "bull_pre_covid":  ("2017-01-01", "2020-02-01"),
    "covid_crash":     ("2020-01-01", "2021-12-31"),
    "bear_inflation":  ("2022-01-01", "2022-12-31"),
    "ai_rebound":      ("2023-01-01", "2024-12-31"),
    "full_10y":        ("2015-01-01", "2024-12-31"),
}

# look_back per timeframe
LOOK_BACK = {"daily": 60, "weekly": 52}


def load_xlsx_file(filepath: str) -> pd.DataFrame:
    """Load an XLSX file and return a cleaned DataFrame with DatetimeIndex."""
    df = pd.read_excel(filepath, engine="openpyxl")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    date_cols = [c for c in df.columns if any(k in c for k in ["date", "time", "datetime", "timestamp"])]
    if date_cols:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
        df = df.set_index(date_cols[0]).sort_index()

    col_map = {}
    for col in df.columns:
        if "close" in col or col == "c":
            col_map["close"] = col
        elif "open" in col or col == "o":
            col_map["open"] = col
        elif "high" in col or col == "h":
            col_map["high"] = col
        elif "low" in col or col == "l":
            col_map["low"] = col
        elif "volume" in col or col == "v":
            col_map["volume"] = col

    df = df.rename(columns={v: k for k, v in col_map.items()})

    if "close" not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 1:
            df = df.rename(columns={numeric_cols[0]: "close"})
        else:
            raise ValueError(f"Cannot identify 'close' column in {filepath}")

    if "high" not in df.columns:
        df["high"] = df["close"]
    if "low" not in df.columns:
        df["low"] = df["close"]

    cols = ["close", "high", "low"] + [c for c in df.columns if c not in ["close", "high", "low"]]
    df = df[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
    return df


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a daily DataFrame to weekly OHLC."""
    agg = {"close": "last", "high": "max", "low": "min"}
    extra = {c: "last" for c in df.columns if c not in agg}
    df_w = df.resample("W").agg({**agg, **extra}).dropna(subset=["close"])
    return df_w


def filter_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Slice DataFrame to the requested market period."""
    start, end = PERIODS[period]
    mask = (df.index >= start) & (df.index <= end)
    return df.loc[mask].copy()


def discover_files(data_dir: str, tickers: list) -> dict:
    """Return {ticker: filepath} for the base daily file per ticker."""
    data_dir = Path(data_dir)
    all_files = list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls"))
    logger.info(f"Found {len(all_files)} xlsx files in {data_dir}")

    result = {}
    daily_aliases = ["_daily", "_1d", "_day"]

    for ticker in tickers:
        found = None
        # Prefer a file explicitly named *daily* (underscore-prefixed to avoid matching "intraday")
        for f in all_files:
            fname = f.stem.lower()
            if ticker.lower() in fname and any(fname.endswith(a) or a in fname for a in daily_aliases):
                found = f
                break
        # Fallback: any file with ticker name
        if found is None:
            for f in all_files:
                if ticker.lower() in f.stem.lower():
                    found = f
                    break
        if found:
            result[ticker] = str(found)
            logger.info(f"  {ticker} -> {found.name}")
        else:
            logger.warning(f"  No file found for {ticker}")
    return result


def load_all_data(data_dir: str, tickers: list, timeframes: list, periods: list) -> dict:
    """
    Load and prepare all data slices.
    Returns: {ticker: {timeframe: {period: pd.DataFrame}}}
    Weekly is derived by resampling from the daily file.
    """
    file_map = discover_files(data_dir, tickers)
    data = {}

    for ticker in tickers:
        fp = file_map.get(ticker)
        if fp is None:
            logger.warning(f"Skipping {ticker}: no file found")
            continue

        try:
            df_daily = load_xlsx_file(fp)
            logger.info(f"Loaded {ticker}: {len(df_daily)} daily rows "
                        f"[{df_daily.index.min().date()} → {df_daily.index.max().date()}]")
        except Exception as e:
            logger.error(f"Failed to load {ticker}: {e}")
            continue

        data[ticker] = {}

        for tf in timeframes:
            if tf == "daily":
                df_base = df_daily
            elif tf == "weekly":
                df_base = resample_weekly(df_daily)
                logger.info(f"  {ticker}/weekly: {len(df_base)} rows after resampling")
            else:
                logger.warning(f"Unknown timeframe '{tf}', skipping")
                continue

            data[ticker][tf] = {}
            for period in periods:
                df_slice = filter_period(df_base, period)
                if len(df_slice) == 0:
                    logger.warning(f"  {ticker}/{tf}/{period}: empty after filtering")
                    continue
                data[ticker][tf][period] = df_slice
                logger.info(f"  {ticker}/{tf}/{period}: {len(df_slice)} rows")

    return data


def create_sequences(series: np.ndarray, look_back: int, horizon: int = 15):
    X, y = [], []
    for i in range(len(series) - look_back - horizon + 1):
        X.append(series[i: i + look_back])
        y.append(series[i + look_back: i + look_back + horizon])
    return np.array(X), np.array(y)


def train_val_test_split(X, y, val_ratio=0.15, test_ratio=0.15):
    """Chronological split — no shuffle."""
    n = len(X)
    test_start = int(n * (1 - test_ratio))
    val_start = int(n * (1 - test_ratio - val_ratio))
    return (X[:val_start], y[:val_start]), (X[val_start:test_start], y[val_start:test_start]), (X[test_start:], y[test_start:])


class DataPipeline:
    def __init__(self, look_back: int = 60, horizon: int = 15):
        self.look_back = look_back
        self.horizon = horizon
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fit_transform(self, series: np.ndarray):
        self.scaler.fit(series.reshape(-1, 1))
        return self.scaler.transform(series.reshape(-1, 1)).flatten()

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        return self.scaler.inverse_transform(arr.reshape(-1, 1)).reshape(shape)

    def prepare(self, series: np.ndarray):
        scaled = self.fit_transform(series)
        X, y = create_sequences(scaled, self.look_back, self.horizon)
        train, val, test = train_val_test_split(X, y)
        return train, val, test, scaled
