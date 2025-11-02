"""Tests for signal quality analysis."""
import numpy as np
import pandas as pd
import pytest

from tenbaggers_detector.analysis import SignalQualityAnalyzer


class TestSignalQualityAnalyzer:
    """Test suite for SignalQualityAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return SignalQualityAnalyzer()

    @pytest.fixture
    def redundant_signals(self):
        """Create signals with obvious redundancy."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')

        # Ticker with many consecutive signals (redundant)
        signals = pd.DataFrame({
            'TICKER1': [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, -1, -1, 0, 0, 0, 1, 1, 1, 0, 0],
            'TICKER2': [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1],
        }, index=dates)

        return signals

    @pytest.fixture
    def signal_returns(self):
        """Create corresponding signal returns."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=20, freq='D')

        # Random returns
        returns = pd.DataFrame({
            'TICKER1': np.random.normal(0.02, 0.05, 20),
            'TICKER2': np.random.normal(0.01, 0.04, 20),
        }, index=dates)

        return returns

    def test_filter_redundant_signals(self, analyzer, redundant_signals):
        """Test redundant signal filtering."""
        lookahead = 5

        filtered = analyzer.filter_redundant_signals(redundant_signals, lookahead)

        # Check shape preserved
        assert filtered.shape == redundant_signals.shape

        # Count signals before/after
        signals_before = (redundant_signals != 0).sum().sum()
        signals_after = (filtered != 0).sum().sum()

        # Should have fewer signals after filtering
        assert signals_after <= signals_before

        # Verify no consecutive signals within lookahead window
        for ticker in filtered.columns:
            ticker_signals = filtered[ticker]
            signal_dates = ticker_signals[ticker_signals != 0].index

            for i in range(len(signal_dates) - 1):
                days_diff = (signal_dates[i + 1] - signal_dates[i]).days
                assert days_diff >= lookahead, \
                    f"Found signals within {lookahead} days for {ticker}"

    def test_clear_signals_long_only(self, analyzer):
        """Test signal clearing for long signals only."""
        # Create series with consecutive 1s
        signals = pd.Series([1, 1, 0, 1, 1, 1, 0, 0, 1, 0])
        window = 3

        cleaned = analyzer._clear_signals(signals, window)

        # First signal should remain
        assert cleaned.iloc[0] == 1

        # No signals within window after first
        assert cleaned.iloc[1] == 0  # Too close to index 0
        assert cleaned.iloc[2] == 0  # No signal anyway

        # Signal at index 3 should remain (3+ days from index 0)
        assert cleaned.iloc[3] == 1

        # Signals at 4, 5 should be removed (too close to index 3)
        assert cleaned.iloc[4] == 0
        assert cleaned.iloc[5] == 0

    def test_calculate_metrics(self, analyzer, redundant_signals, signal_returns):
        """Test signal quality metrics calculation."""
        # Create signal returns (only where signals exist)
        masked_returns = signal_returns.copy()
        masked_returns[redundant_signals == 0] = np.nan

        metrics = analyzer.calculate_metrics(redundant_signals, masked_returns)

        assert metrics.total_signals > 0
        assert metrics.tickers_with_signals > 0
        assert isinstance(metrics.mean_return, float)
        assert isinstance(metrics.std_return, float)
        assert 0 <= metrics.win_rate <= 1

    def test_calculate_metrics_empty(self, analyzer):
        """Test metrics on empty data."""
        empty_signals = pd.DataFrame()
        empty_returns = pd.DataFrame()

        metrics = analyzer.calculate_metrics(empty_signals, empty_returns)

        assert metrics.total_signals == 0
        assert metrics.mean_return == 0.0
        assert metrics.win_rate == 0.0

    def test_compare_before_after_filter(
        self, analyzer, redundant_signals, signal_returns
    ):
        """Test before/after comparison."""
        lookahead = 5

        # Filter signals
        filtered_signals = analyzer.filter_redundant_signals(
            redundant_signals, lookahead
        )

        # Create masked returns
        returns_before = signal_returns.copy()
        returns_before[redundant_signals == 0] = np.nan

        returns_after = signal_returns.copy()
        returns_after[filtered_signals == 0] = np.nan

        # Compare
        metrics_before, metrics_after = analyzer.compare_before_after_filter(
            redundant_signals,
            filtered_signals,
            returns_before,
            returns_after,
        )

        # Should have fewer signals after
        assert metrics_after.total_signals <= metrics_before.total_signals

        # Reduction percentage should be calculated
        if metrics_before.total_signals > 0:
            assert metrics_after.reduction_pct >= 0

    def test_generate_comparison_report(self, analyzer):
        """Test report generation."""
        # Create mock metrics
        from tenbaggers_detector.analysis.signal_quality import SignalQualityMetrics

        metrics_before = SignalQualityMetrics(
            total_signals=100,
            signals_after_filter=100,
            reduction_pct=0.0,
            avg_signals_per_ticker=10.0,
            tickers_with_signals=10,
            mean_return=0.02,
            median_return=0.015,
            std_return=0.05,
            win_rate=0.55,
            signal_to_noise=0.4,
        )

        metrics_after = SignalQualityMetrics(
            total_signals=60,
            signals_after_filter=60,
            reduction_pct=40.0,
            avg_signals_per_ticker=6.0,
            tickers_with_signals=10,
            mean_return=0.025,
            median_return=0.018,
            std_return=0.045,
            win_rate=0.58,
            signal_to_noise=0.55,
        )

        report = analyzer.generate_comparison_report(metrics_before, metrics_after)

        assert isinstance(report, str)
        assert "SIGNAL QUALITY COMPARISON" in report
        assert "40.0%" in report  # Reduction percentage

    def test_win_rate_calculation(self, analyzer):
        """Test win rate calculation."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')

        # Signals: 5 long, 5 short
        signals = pd.DataFrame({
            'TICKER': [1, 1, 1, 1, 1, -1, -1, -1, -1, -1]
        }, index=dates)

        # Returns: 3 positive, 2 negative, repeat
        returns = pd.DataFrame({
            'TICKER': [0.02, 0.01, -0.01, 0.03, -0.02, 0.01, -0.01, 0.02, -0.01, 0.01]
        }, index=dates)

        metrics = analyzer.calculate_metrics(signals, returns)

        # Win rate should be 6/10 = 0.6
        expected_win_rate = 6 / 10
        assert abs(metrics.win_rate - expected_win_rate) < 0.01

    def test_signal_to_noise_ratio(self, analyzer):
        """Test signal-to-noise ratio calculation."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')

        signals = pd.DataFrame({
            'TICKER': [1] * 10
        }, index=dates)

        # Returns with known mean and std
        # Mean = 0.02, Std ≈ 0.01
        returns = pd.DataFrame({
            'TICKER': [0.02, 0.03, 0.01, 0.02, 0.02, 0.03, 0.01, 0.02, 0.03, 0.01]
        }, index=dates)

        metrics = analyzer.calculate_metrics(signals, returns)

        # Signal-to-noise should be mean/std
        expected_snr = 0.02 / returns['TICKER'].std()
        assert abs(metrics.signal_to_noise - expected_snr) < 0.1

    def test_long_short_separation(self, analyzer):
        """Test that long and short signals are filtered separately."""
        dates = pd.date_range('2020-01-01', periods=15, freq='D')

        # Mix of long and short signals
        signals = pd.DataFrame({
            'TICKER': [1, 1, 0, -1, -1, 0, 1, 0, -1, 0, 1, 1, 0, -1, -1]
        }, index=dates)

        lookahead = 3
        filtered = analyzer.filter_redundant_signals(signals, lookahead)

        # Extract filtered long and short separately
        filtered_long = (filtered['TICKER'] == 1)
        filtered_short = (filtered['TICKER'] == -1)

        # Verify spacing within each type
        long_dates = filtered[filtered_long].index
        for i in range(len(long_dates) - 1):
            assert (long_dates[i + 1] - long_dates[i]).days >= lookahead

        short_dates = filtered[filtered_short].index
        for i in range(len(short_dates) - 1):
            assert (short_dates[i + 1] - short_dates[i]).days >= lookahead
