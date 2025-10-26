"""Breakout signal computation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from ..data.models import MarketData, SignalResult, VolumeProfile
from ..features.technical import (
    SMA_WINDOWS,
    add_sma_features,
    bullish_sma_order,
    compute_compression_metrics,
    sma_distance,
    sma_slopes,
    volume_zscore,
)
from ..features.volume_profile import build_volume_profile


@dataclass
class BreakoutConfig:
    lookback_days: int = 1260
    volume_bins: int = 160
    zscore_threshold: float = 2.0
    compression_percentile_max: float = 0.35
    breakout_lookback_high_days: int = 252


@dataclass
class BreakoutFeatures:
    volume_profile: VolumeProfile
    sma_slopes: Dict[int, float]
    volume_zscore: float
    sma_bullish: bool
    volatility_percentile: float
    distances: Dict[str, float]
    breakout_high: float


class BreakoutDetector:
    """Detects breakout candidates following the 10x hunter specification."""

    def __init__(self, config: BreakoutConfig | None = None) -> None:
        self.config = config or BreakoutConfig()

    def evaluate(self, data: MarketData) -> SignalResult | None:
        df = add_sma_features(data.ohlcv)
        df = compute_compression_metrics(df)
        df["vol_z"] = volume_zscore(df)
        volume_profile = build_volume_profile(df, self.config.lookback_days, self.config.volume_bins)
        slopes = sma_slopes(df)
        latest = df.iloc[-1]
        sma_values = {window: float(latest.get(f"sma_{window}")) for window in SMA_WINDOWS}
        compression_ok = latest["rolling_vol_pctl_60"] <= self.config.compression_percentile_max
        breakout_high = df["close"].rolling(self.config.breakout_lookback_high_days).max().iloc[-2]
        breakout_price = float(max(volume_profile.value_area_high, breakout_high))
        breakout_condition = latest["close"] >= breakout_price
        zscore = float(latest["vol_z"])
        zscore_condition = np.isfinite(zscore) and zscore >= self.config.zscore_threshold
        sma_bullish = bullish_sma_order(latest)

        if not (breakout_condition and zscore_condition and compression_ok and sma_bullish):
            return None

        distances = sma_distance(latest)
        notes = []
        if breakout_condition:
            notes.append("Cierre supera VAH/máximo reciente")
        if zscore_condition:
            notes.append(f"Volumen z={zscore:.2f}")
        if compression_ok:
            notes.append("Compresión previa confirmada")
        if sma_bullish:
            notes.append("Estructura de medias alcista")

        score = self._score_candidate(
            volume_profile=volume_profile,
            zscore=zscore,
            sma_slopes=slopes,
            compression=1 - latest["rolling_vol_pctl_60"],
            breakout_strength=latest["close"] / breakout_price - 1,
        )

        return SignalResult(
            ticker=data.ticker,
            date_signal=df.index[-1].to_pydatetime(),
            price_close=float(latest["close"]),
            price_poc=volume_profile.poc_price,
            val=volume_profile.value_area_low,
            vah=volume_profile.value_area_high,
            breakout_level=breakout_price,
            z_volume=zscore,
            sma=sma_values,
            sma_order_bullish=sma_bullish,
            volatility_pctl_60d=float(latest["rolling_vol_pctl_60"]),
            hvn_bands=volume_profile.hvn_bands,
            lvn_bands=volume_profile.lvn_bands,
            score=score,
            notes="; ".join(notes),
        )

    def _score_candidate(
        self,
        *,
        volume_profile: VolumeProfile,
        zscore: float,
        sma_slopes: Dict[int, float],
        compression: float,
        breakout_strength: float,
    ) -> float:
        score = 0.0
        # Accumulation strength
        poc_relative = (volume_profile.poc_price - volume_profile.bin_edges[0]) / (
            volume_profile.bin_edges[-1] - volume_profile.bin_edges[0]
        )
        if poc_relative <= 0.33:
            score += 20
        if volume_profile.value_area_high - volume_profile.value_area_low <= 0.25 * volume_profile.poc_price:
            score += 10

        # SMA structure
        slopes_positive = sum(float(slope > 0) for slope in sma_slopes.values())
        score += 5 * slopes_positive

        # Breakout quality
        score += max(0.0, min(20.0, (zscore - self.config.zscore_threshold) * 10))
        score += max(0.0, min(20.0, breakout_strength * 100))

        # Compression
        score += max(0.0, min(15.0, compression * 15))

        return min(score, 100.0)
