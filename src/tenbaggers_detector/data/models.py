"""Data models for the tenbaggers detector package."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class MarketData:
    """Container for OHLCV data and metadata.

    Attributes
    ----------
    ticker:
        Symbol of the security.
    ohlcv:
        DataFrame with columns `open`, `high`, `low`, `close`, `volume`, and
        optional `adj_close`. The index must be of type ``DatetimeIndex`` and
        sorted ascending.
    metadata:
        Extra metadata like ``timezone``, ``currency`` or API specific
        information.
    universe_alive:
        Boolean Series marking whether the security was part of the trading
        universe on each date. This allows avoiding survivorship bias when
        computing aggregates.
    """

    ticker: str
    ohlcv: pd.DataFrame
    metadata: Dict[str, str] = field(default_factory=dict)
    universe_alive: Optional[pd.Series] = None

    def __post_init__(self) -> None:
        if not isinstance(self.ohlcv.index, pd.DatetimeIndex):
            raise TypeError("ohlcv index must be a DatetimeIndex")
        if not self.ohlcv.index.is_monotonic_increasing:
            self.ohlcv = self.ohlcv.sort_index()
        if self.universe_alive is not None:
            self.universe_alive = self.universe_alive.reindex(self.ohlcv.index, method="ffill").fillna(False)


@dataclass
class VolumeProfile:
    """Stores the result of a volume profile calculation."""

    poc_price: float
    value_area_low: float
    value_area_high: float
    hvn_bands: List[List[float]]
    lvn_bands: List[List[float]]
    bin_edges: List[float]
    bin_volumes: List[float]


@dataclass
class SignalResult:
    """A scored breakout candidate."""

    ticker: str
    date_signal: datetime
    price_close: float
    price_poc: float
    val: float
    vah: float
    breakout_level: float
    z_volume: float
    sma: Dict[int, float]
    sma_order_bullish: bool
    volatility_pctl_60d: float
    hvn_bands: List[List[float]]
    lvn_bands: List[List[float]]
    score: float
    notes: str

    def to_json(self) -> Dict[str, object]:
        """Return a JSON serialisable dictionary."""
        return {
            "ticker": self.ticker,
            "date_signal": self.date_signal.strftime("%Y-%m-%d"),
            "price_close": round(self.price_close, 4),
            "price_poc": round(self.price_poc, 4),
            "val": round(self.val, 4),
            "vah": round(self.vah, 4),
            "breakout_level": round(self.breakout_level, 4),
            "z_volume": round(self.z_volume, 4),
            "sma": {str(k): round(v, 4) for k, v in self.sma.items()},
            "sma_order_bullish": self.sma_order_bullish,
            "volatility_pctl_60d": round(self.volatility_pctl_60d, 4),
            "hvn_bands": [[round(x, 4) for x in band] for band in self.hvn_bands],
            "lvn_bands": [[round(x, 4) for x in band] for band in self.lvn_bands],
            "score": round(self.score, 2),
            "notes": self.notes,
        }
