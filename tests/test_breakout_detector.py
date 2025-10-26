import numpy as np
import pandas as pd

from tenbaggers_detector.data.models import MarketData
from tenbaggers_detector.signals.breakout import BreakoutConfig, BreakoutDetector


def make_test_data():
    dates = pd.date_range("2020-01-01", periods=400)
    close = np.linspace(5, 20, len(dates))
    high = close + 0.2
    low = close - 0.2
    open_ = close - 0.1
    volume = np.full(len(dates), 500_000.0)
    volume[-1] = 1_500_000.0
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    return MarketData(ticker="TEST", ohlcv=df)


def test_breakout_detector_scores_candidate():
    data = make_test_data()
    config = BreakoutConfig(lookback_days=300, volume_bins=50, zscore_threshold=1.5, compression_percentile_max=1.0)
    detector = BreakoutDetector(config)
    result = detector.evaluate(data)
    assert result is not None
    assert result.score >= 70
    assert result.sma_order_bullish is True
