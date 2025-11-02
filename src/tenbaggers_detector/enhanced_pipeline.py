"""Enhanced pipeline with outlier detection and robustness validation.

This pipeline extends the basic TenbaggerPipeline with:
1. Statistical outlier detection using KS test
2. Signal quality analysis and filtering
3. Robustness validation
4. Comprehensive reporting

Usage:
    from tenbaggers_detector.enhanced_pipeline import EnhancedPipeline, EnhancedConfig
    from tenbaggers_detector.data.sources import YFinanceSource

    source = YFinanceSource()
    config = EnhancedConfig(
        enable_outlier_detection=True,
        enable_robustness_validation=True,
    )

    pipeline = EnhancedPipeline(source, config)
    results = pipeline.run(['AAPL', 'MSFT', 'GOOGL'])

    # Print comprehensive analysis
    print(pipeline.get_analysis_report())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set

import numpy as np
import pandas as pd

from .analysis import (
    OutlierConfig,
    OutlierDetector,
    RobustnessValidator,
    SignalQualityAnalyzer,
)
from .data.models import MarketData, SignalResult
from .data.sources import MarketDataSource
from .pipeline import PipelineConfig, TenbaggerPipeline
from .preprocessing import UniverseFilters


@dataclass
class EnhancedConfig:
    """Configuration for enhanced pipeline with validation."""

    # Base pipeline config
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    # Outlier detection
    enable_outlier_detection: bool = True
    outlier_config: OutlierConfig = field(default_factory=OutlierConfig)
    exclude_outliers: bool = True  # Auto-exclude severe outliers

    # Signal quality analysis
    enable_signal_filtering: bool = True
    filter_lookahead_days: int = 5  # Remove redundant signals within N days

    # Robustness validation
    enable_robustness_validation: bool = True
    return_tolerance: float = 0.05  # 5% acceptable difference
    sharpe_tolerance: float = 0.3   # 0.3 Sharpe acceptable difference

    # Reporting
    verbose: bool = True  # Print analysis reports


class EnhancedPipeline:
    """Enhanced pipeline with statistical validation and outlier detection.

    This pipeline wraps the basic TenbaggerPipeline and adds:
    - Outlier detection using Kolmogorov-Smirnov test
    - Signal quality analysis
    - Robustness validation
    - Comprehensive reporting
    """

    def __init__(
        self,
        source: MarketDataSource,
        config: EnhancedConfig | None = None,
    ) -> None:
        """Initialize enhanced pipeline.

        Args:
            source: Market data source
            config: Enhanced configuration (optional)
        """
        self.source = source
        self.config = config or EnhancedConfig()

        # Initialize base pipeline
        self.pipeline = TenbaggerPipeline(source, self.config.pipeline)

        # Initialize analysis modules
        self.outlier_detector = OutlierDetector(self.config.outlier_config)
        self.signal_analyzer = SignalQualityAnalyzer()
        self.robustness_validator = RobustnessValidator(
            return_tolerance=self.config.return_tolerance,
            sharpe_tolerance=self.config.sharpe_tolerance,
        )

        # Store analysis results
        self.outlier_results = None
        self.excluded_tickers: Set[str] = set()
        self.signal_quality_before = None
        self.signal_quality_after = None
        self.robustness_report = None

    def run(
        self,
        tickers: Iterable[str],
        start: str | None = None,
        end: str | None = None,
    ) -> List[SignalResult]:
        """Run enhanced pipeline with validation.

        Args:
            tickers: List of ticker symbols
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)

        Returns:
            List of validated signal results
        """
        ticker_list = list(tickers)

        # Step 1: Run base pipeline to get initial signals
        if self.config.verbose:
            print("\n🔍 Step 1: Running base breakout detection...")

        initial_results = self.pipeline.run(ticker_list, start=start, end=end)

        if not initial_results:
            if self.config.verbose:
                print("❌ No signals detected")
            return []

        if self.config.verbose:
            print(f"✅ Found {len(initial_results)} initial signals")

        # Step 2: Outlier detection (if enabled)
        if self.config.enable_outlier_detection:
            if self.config.verbose:
                print("\n🔍 Step 2: Running outlier detection...")

            self._run_outlier_detection(initial_results)

            if self.config.verbose and self.outlier_results:
                print(self.outlier_detector.generate_report(self.outlier_results))

        # Step 3: Filter signals (if enabled)
        final_results = initial_results

        if self.config.enable_signal_filtering:
            if self.config.verbose:
                print("\n🔍 Step 3: Filtering redundant signals...")

            final_results = self._filter_redundant_signals(initial_results)

        # Step 4: Robustness validation (if enabled)
        if self.config.enable_robustness_validation and self.outlier_results:
            if self.config.verbose:
                print("\n🔍 Step 4: Validating robustness...")

            self._run_robustness_validation(final_results, ticker_list)

            if self.config.verbose and self.robustness_report:
                print(
                    self.robustness_validator.generate_report(self.robustness_report)
                )

        if self.config.verbose:
            print(f"\n✅ Final signals: {len(final_results)}")

        return final_results

    def _run_outlier_detection(self, results: List[SignalResult]) -> None:
        """Run outlier detection on signal results."""
        # Convert results to return matrix for analysis
        # This is a simplified version - in practice, you'd need actual return data
        signal_returns = self._extract_signal_returns(results)

        if signal_returns.empty:
            return

        # Run outlier analysis
        self.outlier_results = self.outlier_detector.analyze(signal_returns)

        # Get exclusion list
        if self.config.exclude_outliers:
            self.excluded_tickers = self.outlier_detector.get_exclusion_list(
                self.outlier_results
            )

    def _filter_redundant_signals(
        self, results: List[SignalResult]
    ) -> List[SignalResult]:
        """Filter redundant signals within lookahead window."""
        # Group by ticker
        ticker_signals: Dict[str, List[SignalResult]] = {}
        for result in results:
            if result.ticker not in ticker_signals:
                ticker_signals[result.ticker] = []
            ticker_signals[result.ticker].append(result)

        # Filter each ticker's signals
        filtered_results = []
        for ticker, signals in ticker_signals.items():
            # Sort by date
            signals_sorted = sorted(signals, key=lambda x: x.date_signal)

            # Keep first signal, then only signals after lookahead window
            if signals_sorted:
                filtered_results.append(signals_sorted[0])
                last_date = signals_sorted[0].date_signal

                for signal in signals_sorted[1:]:
                    days_diff = (signal.date_signal - last_date).days
                    if days_diff >= self.config.filter_lookahead_days:
                        filtered_results.append(signal)
                        last_date = signal.date_signal

        return filtered_results

    def _run_robustness_validation(
        self, results: List[SignalResult], all_tickers: List[str]
    ) -> None:
        """Run robustness validation comparing with/without outliers."""
        # Extract returns
        all_returns = self._extract_returns_series(results, exclude_outliers=False)
        clean_returns = self._extract_returns_series(results, exclude_outliers=True)

        if len(all_returns) == 0 or len(clean_returns) == 0:
            return

        # Run validation
        self.robustness_report = self.robustness_validator.validate(
            returns_full=all_returns,
            returns_clean=clean_returns,
            outlier_tickers=self.excluded_tickers,
            total_tickers=len(all_tickers),
        )

    def _extract_signal_returns(self, results: List[SignalResult]) -> pd.DataFrame:
        """Extract signal returns as DataFrame for outlier analysis.

        Note: This is a simplified implementation. In practice, you'd need
        actual historical returns data to compute proper signal returns.

        For now, we'll use score as a proxy for return quality.
        """
        # Group by ticker and date
        data = []
        for result in results:
            # Use normalized score as proxy for return
            # In real implementation, you'd calculate actual forward returns
            proxy_return = (result.score - 70) / 30  # Normalize score to ~return
            data.append({
                'date': result.date_signal,
                'ticker': result.ticker,
                'signal_return': proxy_return,
            })

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Pivot to wide format (dates x tickers)
        pivot = df.pivot(index='date', columns='ticker', values='signal_return')
        pivot = pivot.fillna(0)  # 0 = no signal

        return pivot

    def _extract_returns_series(
        self, results: List[SignalResult], exclude_outliers: bool
    ) -> pd.Series:
        """Extract returns as Series for robustness validation."""
        returns = []

        for result in results:
            # Skip if outlier and we're excluding them
            if exclude_outliers and result.ticker in self.excluded_tickers:
                continue

            # Use normalized score as proxy
            proxy_return = (result.score - 70) / 30
            returns.append(proxy_return)

        return pd.Series(returns)

    def get_analysis_report(self) -> str:
        """Generate comprehensive analysis report.

        Returns:
            Formatted report with all analysis results
        """
        lines = [
            "=" * 70,
            "ENHANCED PIPELINE ANALYSIS REPORT",
            "=" * 70,
            "",
        ]

        # Outlier analysis
        if self.outlier_results:
            lines.append(self.outlier_detector.generate_report(self.outlier_results))
            lines.append("")

        # Signal quality
        if self.signal_quality_before and self.signal_quality_after:
            lines.append(
                self.signal_analyzer.generate_comparison_report(
                    self.signal_quality_before, self.signal_quality_after
                )
            )
            lines.append("")

        # Robustness validation
        if self.robustness_report:
            lines.append(
                self.robustness_validator.generate_report(self.robustness_report)
            )
            lines.append("")

        return "\n".join(lines)

    def get_clean_universe(self, tickers: List[str]) -> List[str]:
        """Get list of tickers excluding detected outliers.

        Args:
            tickers: Original list of tickers

        Returns:
            Filtered list without outliers
        """
        return [t for t in tickers if t not in self.excluded_tickers]

    def get_excluded_universe(self) -> Set[str]:
        """Get set of excluded ticker symbols.

        Returns:
            Set of outlier tickers
        """
        return self.excluded_tickers.copy()
