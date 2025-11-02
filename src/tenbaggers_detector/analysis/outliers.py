"""Outlier detection using Kolmogorov-Smirnov test.

This module implements statistical outlier detection to identify tickers with
anomalous return distributions that could indicate:
- Data quality issues (splits not adjusted, API errors)
- Corporate events (M&A, bankruptcies, spin-offs)
- Extreme volatility (not representative of normal market behavior)

The goal is to EXCLUDE these outliers from the strategy to prevent:
- Overfitting to anomalous behavior
- Misleading backtest results
- Poor performance in live trading
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from scipy.stats import kstest


@dataclass
class OutlierConfig:
    """Configuration for outlier detection.

    Practical thresholds based on real S&P 500 market data:
    - KS < 0.20: Very normal behavior (ideal)
    - KS 0.20-0.30: Normal with typical volatility (acceptable)
    - KS 0.30-0.40: Moderate outlier (review required)
    - KS > 0.40: Severe outlier (exclude from strategy)
    """

    # Threshold for severe outliers (auto-exclude)
    ks_threshold_severe: float = 0.40

    # Threshold for moderate outliers (flag for review)
    ks_threshold_moderate: float = 0.30

    # P-value threshold for statistical significance
    pvalue_threshold: float = 0.05

    # Minimum number of signals required for KS test
    min_signals: int = 10


@dataclass
class OutlierAnalysisResult:
    """Result of outlier analysis for a single ticker."""

    ticker: str
    ks_statistic: float
    p_value: float
    num_signals: int
    mean_return: float
    std_return: float
    is_severe: bool  # KS > threshold_severe
    is_moderate: bool  # KS > threshold_moderate
    should_exclude: bool  # Recommendation to exclude from strategy

    @property
    def severity_label(self) -> str:
        """Human-readable severity classification."""
        if self.is_severe:
            return "SEVERE"
        elif self.is_moderate:
            return "MODERATE"
        else:
            return "NORMAL"


class OutlierDetector:
    """Detects statistical outliers using Kolmogorov-Smirnov test.

    Usage:
        detector = OutlierDetector()

        # Analyze signal returns DataFrame (columns = tickers, values = returns)
        results = detector.analyze(signal_returns)

        # Get tickers to exclude
        excluded = detector.get_exclusion_list(results)

        # Filter clean universe
        clean_tickers = [t for t in all_tickers if t not in excluded]
    """

    def __init__(self, config: OutlierConfig | None = None) -> None:
        self.config = config or OutlierConfig()

    def analyze(self, signal_returns: pd.DataFrame) -> List[OutlierAnalysisResult]:
        """Analyze signal returns to detect outliers.

        Args:
            signal_returns: DataFrame with signal returns
                - Index: dates
                - Columns: tickers
                - Values: returns (0 = no signal)

        Returns:
            List of OutlierAnalysisResult, sorted by KS statistic (descending)
        """
        # Convert to long format for KS test
        long_format = self._prepare_long_format(signal_returns)

        if len(long_format) < self.config.min_signals:
            return []

        # Calculate global distribution parameters
        overall_mean = long_format['signal_return'].mean()
        overall_std = long_format['signal_return'].std(ddof=0)

        results: List[OutlierAnalysisResult] = []

        # Perform KS test for each ticker
        for ticker, group_data in long_format.groupby('ticker'):
            ticker_returns = group_data['signal_return'].values

            if len(ticker_returns) < self.config.min_signals:
                continue

            # KS test: Compare ticker distribution vs. normal(overall_mean, overall_std)
            ks_stat, p_value = kstest(
                ticker_returns,
                'norm',
                args=(overall_mean, overall_std)
            )

            # Calculate return statistics
            mean_return = ticker_returns.mean()
            std_return = ticker_returns.std(ddof=1)

            # Classify severity
            is_significant = p_value < self.config.pvalue_threshold
            is_severe = ks_stat > self.config.ks_threshold_severe and is_significant
            is_moderate = (
                ks_stat > self.config.ks_threshold_moderate
                and not is_severe
                and is_significant
            )

            # Recommendation: exclude severe outliers
            should_exclude = is_severe

            results.append(OutlierAnalysisResult(
                ticker=ticker,
                ks_statistic=ks_stat,
                p_value=p_value,
                num_signals=len(ticker_returns),
                mean_return=mean_return,
                std_return=std_return,
                is_severe=is_severe,
                is_moderate=is_moderate,
                should_exclude=should_exclude,
            ))

        # Sort by KS statistic (highest first)
        results.sort(key=lambda x: x.ks_statistic, reverse=True)

        return results

    def get_exclusion_list(
        self,
        results: List[OutlierAnalysisResult],
        exclude_moderate: bool = False,
    ) -> Set[str]:
        """Get set of tickers to exclude from strategy.

        Args:
            results: List of outlier analysis results
            exclude_moderate: If True, also exclude moderate outliers

        Returns:
            Set of ticker symbols to exclude
        """
        excluded = set()

        for result in results:
            if result.should_exclude:
                excluded.add(result.ticker)
            elif exclude_moderate and result.is_moderate:
                excluded.add(result.ticker)

        return excluded

    def get_summary_stats(self, results: List[OutlierAnalysisResult]) -> Dict[str, float]:
        """Get summary statistics of outlier analysis.

        Returns:
            Dict with summary metrics
        """
        if not results:
            return {}

        ks_values = [r.ks_statistic for r in results]

        return {
            'total_tickers': len(results),
            'severe_outliers': sum(r.is_severe for r in results),
            'moderate_outliers': sum(r.is_moderate for r in results),
            'normal_tickers': sum(not r.is_severe and not r.is_moderate for r in results),
            'ks_mean': np.mean(ks_values),
            'ks_median': np.median(ks_values),
            'ks_p75': np.percentile(ks_values, 75),
            'ks_p90': np.percentile(ks_values, 90),
            'ks_p95': np.percentile(ks_values, 95),
            'ks_max': np.max(ks_values),
        }

    def _prepare_long_format(self, signal_returns: pd.DataFrame) -> pd.DataFrame:
        """Convert signal returns to long format for KS test.

        Args:
            signal_returns: Wide format DataFrame (dates x tickers)

        Returns:
            Long format DataFrame with columns: ticker, signal_return
        """
        # Filter out zero returns (no signal)
        signal_returns_filtered = signal_returns.copy()
        signal_returns_filtered[signal_returns_filtered == 0] = np.nan

        # Stack to long format
        long_format = signal_returns_filtered.stack().reset_index()
        long_format.columns = ['date', 'ticker', 'signal_return']

        # Drop NaN values
        long_format = long_format.dropna(subset=['signal_return'])

        return long_format[['ticker', 'signal_return']]

    def generate_report(self, results: List[OutlierAnalysisResult]) -> str:
        """Generate human-readable outlier analysis report.

        Args:
            results: List of outlier analysis results

        Returns:
            Formatted report string
        """
        if not results:
            return "No outlier analysis results available."

        stats = self.get_summary_stats(results)
        excluded = self.get_exclusion_list(results)

        report_lines = [
            "=" * 70,
            "OUTLIER ANALYSIS REPORT",
            "=" * 70,
            "",
            "📊 SUMMARY STATISTICS:",
            f"   Total tickers analyzed: {stats['total_tickers']}",
            f"   Severe outliers (KS > {self.config.ks_threshold_severe}): {stats['severe_outliers']}",
            f"   Moderate outliers (KS {self.config.ks_threshold_moderate}-{self.config.ks_threshold_severe}): {stats['moderate_outliers']}",
            f"   Normal tickers: {stats['normal_tickers']}",
            "",
            f"   KS Statistics:",
            f"      Mean: {stats['ks_mean']:.4f}",
            f"      Median: {stats['ks_median']:.4f}",
            f"      P75: {stats['ks_p75']:.4f}",
            f"      P90: {stats['ks_p90']:.4f}",
            f"      P95: {stats['ks_p95']:.4f}",
            f"      Max: {stats['ks_max']:.4f}",
            "",
            f"❌ EXCLUSION LIST ({len(excluded)} tickers):",
        ]

        if excluded:
            report_lines.append(f"   {', '.join(sorted(excluded))}")
        else:
            report_lines.append("   ✅ No severe outliers - excellent!")

        report_lines.extend([
            "",
            "🔝 TOP 10 OUTLIERS BY KS STATISTIC:",
            "",
        ])

        # Table header
        report_lines.append(
            f"{'Ticker':<8} {'KS Value':>10} {'P-Value':>10} {'Signals':>8} {'Severity':<10}"
        )
        report_lines.append("-" * 70)

        # Top 10 outliers
        for result in results[:10]:
            report_lines.append(
                f"{result.ticker:<8} {result.ks_statistic:>10.4f} {result.p_value:>10.4e} "
                f"{result.num_signals:>8} {result.severity_label:<10}"
            )

        report_lines.append("")
        report_lines.append("=" * 70)

        return "\n".join(report_lines)
