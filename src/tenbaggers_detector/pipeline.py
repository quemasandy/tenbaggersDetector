"""High level pipeline orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .data.models import MarketData, SignalResult
from .data.sources import MarketDataSource
from .preprocessing import UniverseFilters, build_universe
from .signals.breakout import BreakoutConfig, BreakoutDetector


@dataclass
class PipelineConfig:
    universe_filters: UniverseFilters = UniverseFilters()
    breakout: BreakoutConfig = BreakoutConfig()


class TenbaggerPipeline:
    """Pipeline tying together data source, preprocessing and signal detection."""

    def __init__(self, source: MarketDataSource, config: PipelineConfig | None = None) -> None:
        self.source = source
        self.config = config or PipelineConfig()
        self.detector = BreakoutDetector(self.config.breakout)

    def run(self, tickers: Iterable[str], start: str | None = None, end: str | None = None) -> List[SignalResult]:
        raw_data = self.source.fetch(tickers, start=start, end=end)
        universe = build_universe(raw_data, self.config.universe_filters)
        results: List[SignalResult] = []
        for item in universe:
            signal = self.detector.evaluate(item)
            if signal and signal.score >= 70:
                results.append(signal)
        return results
