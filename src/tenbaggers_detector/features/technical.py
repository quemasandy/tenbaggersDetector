"""Technical feature engineering for tenbagger candidates."""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import linregress

from ..preprocessing import compute_atr, rolling_volatility

SMA_WINDOWS = (20, 50, 100, 200)


def add_sma_features(df: pd.DataFrame, windows: Iterable[int] = SMA_WINDOWS) -> pd.DataFrame:
    df = df.copy()
    for window in windows:
        df[f"sma_{window}"] = df["close"].rolling(window, min_periods=window).mean()
    return df


def sma_slopes(df: pd.DataFrame, windows: Iterable[int] = SMA_WINDOWS) -> Dict[int, float]:
    slopes: Dict[int, float] = {}
    for window in windows:
        sma_col = f"sma_{window}"
        if sma_col not in df.columns:
            raise KeyError(f"Missing {sma_col} column")
        series = df[sma_col].dropna().tail(window)
        if len(series) < window:
            slopes[window] = np.nan
            continue
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series.values)
        slopes[window] = slope
    return slopes


def volume_zscore(df: pd.DataFrame, window: int = 60) -> pd.Series:
    mean = df["volume"].rolling(window, min_periods=window//2).mean()
    std = df["volume"].rolling(window, min_periods=window//2).std(ddof=0)
    return (df["volume"] - mean) / std


def compute_compression_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr_14"] = compute_atr(df)
    df["rolling_vol_60"] = rolling_volatility(df["close"], 60)
    df["rolling_vol_pctl_60"] = df["rolling_vol_60"].expanding().apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    return df


def bullish_sma_order(row: pd.Series) -> bool:
    sma_values = [row.get(f"sma_{window}") for window in SMA_WINDOWS]
    if any(pd.isna(v) for v in sma_values):
        return False
    return all(earlier > later for earlier, later in zip(sma_values, sma_values[1:]))


def sma_distance(row: pd.Series) -> Dict[str, float]:
    distances = {
        "close_sma200": row["close"] / row.get("sma_200", np.nan) - 1.0,
        "sma20_sma50": row.get("sma_20", np.nan) / row.get("sma_50", np.nan) - 1.0,
    }
    return distances
