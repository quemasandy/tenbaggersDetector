"""Data source abstractions for loading market data."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .models import MarketData


class MarketDataSource:
    """Abstract base class for market data providers."""

    def fetch(self, tickers: Iterable[str], start: Optional[str] = None, end: Optional[str] = None) -> List[MarketData]:
        raise NotImplementedError


@dataclass
class YFinanceSource(MarketDataSource):
    """Fetch OHLCV data using :mod:`yfinance`.

    Parameters
    ----------
    auto_adjust:
        Whether to use adjusted prices. Defaults to ``True``.
    interval:
        Price interval supported by Yahoo! Finance. ``1d`` by default.
    tz:
        Optional timezone metadata to attach.
    """

    auto_adjust: bool = True
    interval: str = "1d"
    tz: Optional[str] = "America/New_York"

    def fetch(self, tickers: Iterable[str], start: Optional[str] = None, end: Optional[str] = None) -> List[MarketData]:
        try:
            import yfinance as yf  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency optional
            raise ImportError("yfinance is required for YFinanceSource") from exc

        tickers = list(tickers)
        data: List[MarketData] = []
        for ticker in tickers:
            history = yf.download(ticker, start=start, end=end, interval=self.interval, auto_adjust=self.auto_adjust, progress=False)
            if history.empty:
                continue
            history.index = pd.to_datetime(history.index)
            history = history.rename(columns=str.lower)
            history = history[[c for c in ["open", "high", "low", "close", "adj close", "volume"] if c in history.columns]]
            if "adj close" in history.columns and "adj_close" not in history.columns:
                history = history.rename(columns={"adj close": "adj_close"})
            metadata = {
                "source": "yfinance",
                "symbol": ticker,
                "timezone": self.tz or "",
                "auto_adjust": json.dumps(self.auto_adjust),
                "interval": self.interval,
                "download_start": str(start) if start else "",
                "download_end": str(end) if end else "",
            }
            data.append(MarketData(ticker=ticker, ohlcv=history, metadata=metadata))
        return data


@dataclass
class CSVSource(MarketDataSource):
    """Load OHLCV data from CSV files on disk."""

    path_template: str
    timezone: Optional[str] = None

    def fetch(self, tickers: Iterable[str], start: Optional[str] = None, end: Optional[str] = None) -> List[MarketData]:
        data: List[MarketData] = []
        for ticker in tickers:
            path = Path(self.path_template.format(ticker=ticker))
            if not path.exists():
                continue
            df = pd.read_csv(path, parse_dates=["date"], index_col="date")
            if start:
                df = df[df.index >= pd.to_datetime(start)]
            if end:
                df = df[df.index <= pd.to_datetime(end)]
            df = df.rename(columns=str.lower)
            data.append(
                MarketData(
                    ticker=ticker,
                    ohlcv=df,
                    metadata={
                        "source": "csv",
                        "path": str(path),
                        "timezone": self.timezone or "",
                    },
                )
            )
        return data
