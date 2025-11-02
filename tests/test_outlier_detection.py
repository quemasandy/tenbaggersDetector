"""Tests for outlier detection module."""
import numpy as np
import pandas as pd
import pytest

from tenbaggers_detector.analysis import OutlierConfig, OutlierDetector


class TestOutlierDetector:
    """Test suite for OutlierDetector."""

    @pytest.fixture
    def detector(self):
        """Create detector with default config."""
        return OutlierDetector()

    @pytest.fixture
    def normal_returns(self):
        """Generate normal signal returns (no outliers)."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        # 5 tickers with normal distribution
        data = {}
        for i in range(5):
            ticker = f"TICKER{i}"
            # Random returns from normal distribution
            returns = np.random.normal(0.01, 0.05, size=100)
            # Add some zeros (no signal days)
            mask = np.random.random(100) > 0.3  # 30% signal days
            returns[~mask] = 0
            data[ticker] = returns

        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def outlier_returns(self):
        """Generate returns with clear outlier."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        data = {}

        # 4 normal tickers
        for i in range(4):
            ticker = f"NORMAL{i}"
            returns = np.random.normal(0.01, 0.05, size=100)
            mask = np.random.random(100) > 0.3
            returns[~mask] = 0
            data[ticker] = returns

        # 1 severe outlier with different distribution
        outlier_returns = np.random.normal(0.10, 0.20, size=100)  # Much higher mean/std
        mask = np.random.random(100) > 0.3
        outlier_returns[~mask] = 0
        data["OUTLIER"] = outlier_returns

        return pd.DataFrame(data, index=dates)

    def test_analyze_normal_returns(self, detector, normal_returns):
        """Test analysis on normal returns (should find no severe outliers)."""
        results = detector.analyze(normal_returns)

        assert len(results) > 0, "Should return results for all tickers"

        # Check that no severe outliers detected
        severe_count = sum(r.is_severe for r in results)
        assert severe_count == 0, "Should not detect severe outliers in normal data"

    def test_analyze_with_outlier(self, detector, outlier_returns):
        """Test analysis detects clear outlier."""
        results = detector.analyze(outlier_returns)

        assert len(results) > 0, "Should return results"

        # Find the outlier result
        outlier_result = None
        for r in results:
            if r.ticker == "OUTLIER":
                outlier_result = r
                break

        assert outlier_result is not None, "Should analyze OUTLIER ticker"

        # The outlier should have higher KS statistic than normal tickers
        normal_ks_values = [r.ks_statistic for r in results if r.ticker != "OUTLIER"]
        max_normal_ks = max(normal_ks_values)

        assert outlier_result.ks_statistic > max_normal_ks, \
            "Outlier should have higher KS than normal tickers"

    def test_get_exclusion_list(self, detector, outlier_returns):
        """Test exclusion list generation."""
        # Use very strict threshold to catch severe outlier
        detector.config.ks_threshold_severe = 0.20

        results = detector.analyze(outlier_returns)
        excluded = detector.get_exclusion_list(results)

        # Should exclude at least the severe outlier
        assert len(excluded) > 0, "Should exclude some tickers with strict threshold"

    def test_get_summary_stats(self, detector, normal_returns):
        """Test summary statistics calculation."""
        results = detector.analyze(normal_returns)
        stats = detector.get_summary_stats(results)

        assert 'total_tickers' in stats
        assert 'severe_outliers' in stats
        assert 'ks_mean' in stats
        assert 'ks_median' in stats

        assert stats['total_tickers'] == len(normal_returns.columns)

    def test_generate_report(self, detector, normal_returns):
        """Test report generation."""
        results = detector.analyze(normal_returns)
        report = detector.generate_report(results)

        assert isinstance(report, str)
        assert len(report) > 0
        assert "OUTLIER ANALYSIS REPORT" in report

    def test_minimum_signals_filter(self, detector):
        """Test that tickers with too few signals are filtered."""
        # Create data with one ticker having very few signals
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = {
            'TICKER1': np.random.normal(0.01, 0.05, 100),  # Many signals
            'TICKER2': np.concatenate([np.random.normal(0.01, 0.05, 5), np.zeros(95)])  # Few signals
        }
        returns = pd.DataFrame(data, index=dates)

        # Set minimum to 10 signals
        detector.config.min_signals = 10

        results = detector.analyze(returns)

        # TICKER2 should be filtered out
        tickers_analyzed = {r.ticker for r in results}
        assert 'TICKER1' in tickers_analyzed
        assert 'TICKER2' not in tickers_analyzed

    def test_custom_config(self):
        """Test detector with custom configuration."""
        custom_config = OutlierConfig(
            ks_threshold_severe=0.50,
            ks_threshold_moderate=0.35,
            pvalue_threshold=0.01,
            min_signals=5,
        )

        detector = OutlierDetector(custom_config)

        assert detector.config.ks_threshold_severe == 0.50
        assert detector.config.ks_threshold_moderate == 0.35
        assert detector.config.pvalue_threshold == 0.01
        assert detector.config.min_signals == 5

    def test_empty_returns(self, detector):
        """Test handling of empty returns DataFrame."""
        empty_df = pd.DataFrame()
        results = detector.analyze(empty_df)

        assert len(results) == 0, "Should return empty list for empty input"

    def test_all_zero_returns(self, detector):
        """Test handling of all-zero returns (no signals)."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = {
            'TICKER1': np.zeros(100),
            'TICKER2': np.zeros(100),
        }
        returns = pd.DataFrame(data, index=dates)

        results = detector.analyze(returns)

        # Should return empty or filter out these tickers
        assert len(results) == 0, "Should not analyze tickers with no signals"
