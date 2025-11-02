"""Signal quality analysis and redundant signal filtering.

This module implements advanced signal filtering to:
- Remove redundant signals within holding periods
- Analyze signal quality and distribution
- Calculate signal-to-noise metrics
- Validate signal robustness
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class SignalQualityMetrics:
    """Metrics describing signal quality."""

    total_signals: int
    signals_after_filter: int
    reduction_pct: float
    avg_signals_per_ticker: float
    tickers_with_signals: int
    mean_return: float
    median_return: float
    std_return: float
    win_rate: float  # Fraction of positive returns
    signal_to_noise: float  # mean / std


class SignalQualityAnalyzer:
    """Analyzes and filters trading signals for quality.

    Main functions:
    1. Filter redundant signals within lookahead windows
    2. Calculate signal quality metrics
    3. Identify high-quality vs low-quality signals
    """

    def filter_redundant_signals(
        self,
        signals: pd.DataFrame,
        lookahead_days: int,
    ) -> pd.DataFrame:
        """Filter redundant signals within lookahead window.

        This implements the same logic as the notebook:
        - For each ticker, ensure only 1 signal per lookahead_days window
        - Separate long (1) and short (-1) signals
        - Apply temporal filtering to each

        Args:
            signals: DataFrame of signals (dates x tickers)
                Values: -1 (short), 0 (no signal), 1 (long)
            lookahead_days: Holding period in days

        Returns:
            Filtered signals DataFrame with same shape
        """
        filtered_signal = signals.copy()

        for ticker in signals.columns:
            ticker_signals = signals[ticker]

            # Separate long and short signals
            long_signals = (ticker_signals == 1).astype(int)
            short_signals = (ticker_signals == -1).astype(int)

            # Clear redundant signals
            filtered_long = self._clear_signals(long_signals, lookahead_days)
            filtered_short = self._clear_signals(short_signals, lookahead_days)

            # Reconstruct combined signal
            filtered_signal[ticker] = filtered_long - filtered_short

        return filtered_signal

    def _clear_signals(self, signals: pd.Series, window_size: int) -> pd.Series:
        """Clear redundant signals in a single-direction series.

        Reduce number of signals to 1 within window_size period.

        A signal is kept only if there are NO signals in the previous
        window_size positions (exclusive of current position).

        Args:
            signals: Series of binary signals (0 or 1)
            window_size: Number of days to have single signal

        Returns:
            Cleaned series with redundant signals removed
        """
        clean_signals = []

        for i, current_signal in enumerate(signals):
            if current_signal == 0:
                # No signal, keep as 0
                clean_signals.append(0)
            else:
                # Check if there's a signal in the last window_size-1 positions
                # For window_size=3, we want signals at least 3 apart:
                # i=0 allowed, i=3 allowed (3-0=3), i=6 allowed (6-3=3)
                # So at i=3, we check positions [1,2] (not [0,1,2])
                start = max(0, i - window_size + 1)
                end = i  # Exclusive, so we don't include current position

                # Check the cleaned signals (what we've already decided to keep)
                has_recent_signal = any(clean_signals[start:end])

                if has_recent_signal:
                    # Too close to a previous signal, discard
                    clean_signals.append(0)
                else:
                    # No recent signal, keep this one
                    clean_signals.append(1)

        return pd.Series(clean_signals, index=signals.index)

    def calculate_metrics(
        self,
        signals: pd.DataFrame,
        signal_returns: pd.DataFrame,
    ) -> SignalQualityMetrics:
        """Calculate quality metrics for signals.

        Args:
            signals: DataFrame of signals (dates x tickers)
            signal_returns: DataFrame of signal returns (dates x tickers)

        Returns:
            SignalQualityMetrics with comprehensive quality analysis
        """
        # Count signals
        total_signals = (signals != 0).sum().sum()

        # Extract non-zero returns
        signal_mask = signals != 0
        active_returns = signal_returns[signal_mask].values.flatten()
        active_returns = active_returns[~np.isnan(active_returns)]

        if len(active_returns) == 0:
            return SignalQualityMetrics(
                total_signals=0,
                signals_after_filter=0,
                reduction_pct=0.0,
                avg_signals_per_ticker=0.0,
                tickers_with_signals=0,
                mean_return=0.0,
                median_return=0.0,
                std_return=0.0,
                win_rate=0.0,
                signal_to_noise=0.0,
            )

        # Calculate metrics
        mean_return = float(np.mean(active_returns))
        median_return = float(np.median(active_returns))
        std_return = float(np.std(active_returns, ddof=1))

        # Win rate (fraction of positive returns)
        win_rate = float(np.sum(active_returns > 0) / len(active_returns))

        # Signal-to-noise ratio
        signal_to_noise = mean_return / std_return if std_return > 0 else 0.0

        # Ticker-level stats
        signals_per_ticker = (signals != 0).sum(axis=0)
        tickers_with_signals = (signals_per_ticker > 0).sum()
        avg_signals_per_ticker = float(signals_per_ticker.mean())

        return SignalQualityMetrics(
            total_signals=int(total_signals),
            signals_after_filter=int(total_signals),  # Same for now
            reduction_pct=0.0,
            avg_signals_per_ticker=avg_signals_per_ticker,
            tickers_with_signals=int(tickers_with_signals),
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            win_rate=win_rate,
            signal_to_noise=signal_to_noise,
        )

    def compare_before_after_filter(
        self,
        signals_before: pd.DataFrame,
        signals_after: pd.DataFrame,
        returns_before: pd.DataFrame,
        returns_after: pd.DataFrame,
    ) -> tuple[SignalQualityMetrics, SignalQualityMetrics]:
        """Compare signal quality before and after filtering.

        Args:
            signals_before: Unfiltered signals
            signals_after: Filtered signals
            returns_before: Returns for unfiltered signals
            returns_after: Returns for filtered signals

        Returns:
            Tuple of (metrics_before, metrics_after)
        """
        metrics_before = self.calculate_metrics(signals_before, returns_before)
        metrics_after = self.calculate_metrics(signals_after, returns_after)

        # Calculate reduction percentage
        if metrics_before.total_signals > 0:
            reduction = (
                1 - metrics_after.total_signals / metrics_before.total_signals
            ) * 100
            metrics_after.reduction_pct = reduction

        return metrics_before, metrics_after

    def generate_comparison_report(
        self,
        metrics_before: SignalQualityMetrics,
        metrics_after: SignalQualityMetrics,
    ) -> str:
        """Generate report comparing signal quality before/after filtering.

        Args:
            metrics_before: Metrics before filtering
            metrics_after: Metrics after filtering

        Returns:
            Formatted comparison report
        """
        report_lines = [
            "=" * 70,
            "SIGNAL QUALITY COMPARISON",
            "=" * 70,
            "",
            "📊 SIGNAL COUNT:",
            f"   Before filtering: {metrics_before.total_signals:,}",
            f"   After filtering:  {metrics_after.total_signals:,}",
            f"   Reduction: {metrics_after.reduction_pct:.1f}%",
            "",
            "📈 RETURN STATISTICS:",
            "",
            f"                  Before         After        Change",
            "-" * 70,
            f"Mean Return:      {metrics_before.mean_return:>8.4f}     {metrics_after.mean_return:>8.4f}     "
            f"{(metrics_after.mean_return - metrics_before.mean_return):>+8.4f}",
            f"Median Return:    {metrics_before.median_return:>8.4f}     {metrics_after.median_return:>8.4f}     "
            f"{(metrics_after.median_return - metrics_before.median_return):>+8.4f}",
            f"Std Deviation:    {metrics_before.std_return:>8.4f}     {metrics_after.std_return:>8.4f}     "
            f"{(metrics_after.std_return - metrics_before.std_return):>+8.4f}",
            f"Win Rate:         {metrics_before.win_rate:>8.2%}     {metrics_after.win_rate:>8.2%}     "
            f"{(metrics_after.win_rate - metrics_before.win_rate):>+8.2%}",
            f"Signal/Noise:     {metrics_before.signal_to_noise:>8.4f}     {metrics_after.signal_to_noise:>8.4f}     "
            f"{(metrics_after.signal_to_noise - metrics_before.signal_to_noise):>+8.4f}",
            "",
            "✅ QUALITY ASSESSMENT:",
        ]

        # Quality assessment
        assessments: List[str] = []

        if metrics_after.mean_return > metrics_before.mean_return:
            assessments.append("   ✅ Mean return IMPROVED after filtering")
        elif metrics_after.mean_return >= metrics_before.mean_return * 0.95:
            assessments.append("   ✅ Mean return STABLE after filtering")
        else:
            assessments.append("   ⚠️  Mean return DECLINED after filtering")

        if metrics_after.signal_to_noise > metrics_before.signal_to_noise:
            assessments.append("   ✅ Signal-to-noise ratio IMPROVED")
        else:
            assessments.append("   ⚠️  Signal-to-noise ratio declined")

        if metrics_after.win_rate >= 0.52:
            assessments.append(f"   ✅ Strong win rate: {metrics_after.win_rate:.1%}")
        elif metrics_after.win_rate >= 0.48:
            assessments.append(f"   ⚠️  Moderate win rate: {metrics_after.win_rate:.1%}")
        else:
            assessments.append(f"   ❌ Low win rate: {metrics_after.win_rate:.1%}")

        report_lines.extend(assessments)
        report_lines.append("")
        report_lines.append("=" * 70)

        return "\n".join(report_lines)
