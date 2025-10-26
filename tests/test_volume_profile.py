import pandas as pd

from tenbaggers_detector.features.volume_profile import build_volume_profile


def test_volume_profile_basic():
    dates = pd.date_range("2020-01-01", periods=5)
    df = pd.DataFrame(
        {
            "open": [10, 10.5, 11, 11.5, 12],
            "high": [10.5, 11, 11.5, 12, 12.5],
            "low": [9.5, 10, 10.5, 11, 11.5],
            "close": [10.2, 10.8, 11.2, 11.8, 12.2],
            "volume": [1_000_000, 1_200_000, 1_100_000, 900_000, 1_300_000],
        },
        index=dates,
    )

    profile = build_volume_profile(df, lookback_days=5, bins=10)

    assert profile.poc_price >= profile.value_area_low
    assert profile.value_area_high > profile.value_area_low
    assert len(profile.bin_edges) == 11
    assert len(profile.bin_volumes) == 10
