"""Preprocessing helpers for the tenbaggers detector."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd

from .data.models import MarketData


@dataclass
class UniverseFilters:
    """Parameters used to build the investable universe."""

    max_price: float = 40.0
    min_dollar_volume: float = 1_000_000.0
    min_volume: float = 300_000.0
    min_history_years: int = 3


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean OHLCV data by removing duplicates, filling holidays and ensuring order."""
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df = df.rename(columns=str.lower)
    required_cols = ["open", "high", "low", "close", "volume"]
    for column in required_cols:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")
    df = df.astype({c: float for c in ["open", "high", "low", "close"]})
    df["volume"] = df["volume"].astype(float)
    return df


def build_universe(data: Iterable[MarketData], filters: UniverseFilters) -> List[MarketData]:
    """Apply liquidity and price filters to a collection of :class:`MarketData`."""
    filtered: List[MarketData] = []
    min_days = filters.min_history_years * 252

    for item in data:
        df = clean_ohlcv(item.ohlcv.copy())
        if len(df) < min_days:
            continue
        df["dollar_volume"] = df["close"] * df["volume"]
        df["adv_60"] = df["dollar_volume"].rolling(60, min_periods=20).mean()
        df["vol_60"] = df["volume"].rolling(60, min_periods=20).mean()
        alive = (
            (df["close"] <= filters.max_price)
            & (df["adv_60"] >= filters.min_dollar_volume)
            & (df["vol_60"] >= filters.min_volume)
        )
        if not alive.any():
            continue
        item.ohlcv = df
        item.universe_alive = alive
        filtered.append(item)
    return filtered


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute the Average True Range."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(period, min_periods=period).mean()
    return atr


def rolling_volatility(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling standard deviation of log returns."""
    log_returns = np.log(series / series.shift())
    return log_returns.rolling(window, min_periods=window//2).std()
