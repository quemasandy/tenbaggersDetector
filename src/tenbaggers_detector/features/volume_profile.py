"""Visible range volume profile approximation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..data.models import VolumeProfile


def build_volume_profile(df: pd.DataFrame, lookback_days: int = 1260, bins: int = 160) -> VolumeProfile:
    if len(df) == 0:
        raise ValueError("Empty dataframe")
    window = df.tail(lookback_days) if lookback_days else df
    price_min = window["low"].min()
    price_max = window["high"].max()
    if np.isclose(price_max, price_min):
        price_max = price_min * 1.01
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_volumes = np.zeros(bins)

    for _, row in window.iterrows():
        lo, hi, vol = row["low"], row["high"], row["volume"]
        if vol == 0 or np.isnan(vol):
            continue
        mask = (bin_edges[:-1] < hi) & (bin_edges[1:] > lo)
        if not mask.any():
            idx = np.searchsorted(bin_edges, row["close"], side="right") - 1
            idx = np.clip(idx, 0, bins - 1)
            bin_volumes[idx] += vol
            continue
        overlaps = np.minimum(bin_edges[1:], hi) - np.maximum(bin_edges[:-1], lo)
        overlaps = np.clip(overlaps, 0, None)
        weights = overlaps[mask]
        total = weights.sum()
        if total == 0:
            continue
        bin_volumes[mask] += vol * weights / total

    poc_idx = int(bin_volumes.argmax())
    poc_price = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2)

    total_volume = bin_volumes.sum()
    order = np.argsort(bin_volumes)[::-1]
    cumulative = 0.0
    selected = np.zeros_like(bin_volumes, dtype=bool)
    for idx in order:
        selected[idx] = True
        cumulative += bin_volumes[idx]
        if cumulative >= 0.7 * total_volume:
            break
    selected_indices = np.where(selected)[0]
    value_area_low = float(bin_edges[selected_indices.min()])
    value_area_high = float(bin_edges[selected_indices.max() + 1])

    hvn_bands, lvn_bands = _detect_peaks_and_valleys(bin_volumes, bin_edges)

    return VolumeProfile(
        poc_price=poc_price,
        value_area_low=value_area_low,
        value_area_high=value_area_high,
        hvn_bands=hvn_bands,
        lvn_bands=lvn_bands,
        bin_edges=bin_edges.tolist(),
        bin_volumes=bin_volumes.tolist(),
    )


def _detect_peaks_and_valleys(volumes: np.ndarray, edges: np.ndarray) -> Tuple[List[List[float]], List[List[float]]]:
    hvns: List[List[float]] = []
    lvns: List[List[float]] = []
    for i in range(1, len(volumes) - 1):
        left, mid, right = volumes[i - 1], volumes[i], volumes[i + 1]
        band = [float(edges[i]), float(edges[i + 1])]
        if mid >= left and mid >= right:
            hvns.append(band)
        if mid <= left and mid <= right:
            lvns.append(band)
    return hvns, lvns
