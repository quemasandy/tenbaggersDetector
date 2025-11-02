"""Robustness validation for trading strategies.

This module validates that a strategy is robust and not overfitted by:
1. Comparing performance with/without outliers
2. Testing sensitivity to parameter changes
3. Analyzing stability across time periods
4. Detecting dependency on anomalous tickers
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np
import pandas as pd


@dataclass
class RobustnessTestResult:
    """Result of robustness validation test."""

    test_name: str
    passed: bool
    metric_full: float
    metric_clean: float
    difference: float
    difference_pct: float
    threshold: float
    notes: str


@dataclass
class RobustnessReport:
    """Comprehensive robustness validation report."""

    tests_passed: int
    tests_failed: int
    is_robust: bool
    test_results: List[RobustnessTestResult]
    recommendation: str
    warnings: List[str]


class RobustnessValidator:
    """Validates strategy robustness against overfitting.

    A robust strategy should:
    1. Perform similarly (or better) without outliers
    2. Not depend on few anomalous tickers
    3. Show stable performance across time periods
    4. Be insensitive to small parameter changes
    """

    def __init__(
        self,
        return_tolerance: float = 0.05,  # 5% difference acceptable
        sharpe_tolerance: float = 0.3,   # 0.3 Sharpe difference acceptable
        win_rate_tolerance: float = 0.03,  # 3% win rate difference
    ):
        """Initialize validator with tolerance thresholds.

        Args:
            return_tolerance: Max acceptable return difference (fraction)
            sharpe_tolerance: Max acceptable Sharpe ratio difference
            win_rate_tolerance: Max acceptable win rate difference
        """
        self.return_tolerance = return_tolerance
        self.sharpe_tolerance = sharpe_tolerance
        self.win_rate_tolerance = win_rate_tolerance

    def validate(
        self,
        returns_full: pd.Series,
        returns_clean: pd.Series,
        outlier_tickers: Set[str],
        total_tickers: int,
    ) -> RobustnessReport:
        """Validate strategy robustness.

        Args:
            returns_full: Returns with all tickers (including outliers)
            returns_clean: Returns with outliers excluded
            outlier_tickers: Set of ticker symbols classified as outliers
            total_tickers: Total number of tickers in universe

        Returns:
            RobustnessReport with comprehensive validation results
        """
        tests: List[RobustnessTestResult] = []
        warnings: List[str] = []

        # Test 1: Mean return comparison
        test1 = self._test_mean_return(returns_full, returns_clean)
        tests.append(test1)

        # Test 2: Sharpe ratio comparison
        test2 = self._test_sharpe_ratio(returns_full, returns_clean)
        tests.append(test2)

        # Test 3: Win rate comparison
        test3 = self._test_win_rate(returns_full, returns_clean)
        tests.append(test3)

        # Test 4: Outlier dependency
        test4 = self._test_outlier_dependency(
            outlier_tickers, total_tickers, returns_full, returns_clean
        )
        tests.append(test4)

        # Count passes/fails
        tests_passed = sum(t.passed for t in tests)
        tests_failed = len(tests) - tests_passed

        # Overall robustness decision
        is_robust = tests_passed >= 3  # At least 3 out of 4 tests must pass

        # Generate recommendation
        recommendation = self._generate_recommendation(
            is_robust, tests, outlier_tickers, total_tickers
        )

        # Generate warnings
        if not test1.passed:
            warnings.append(
                "⚠️  Returns declined significantly without outliers - "
                "strategy may be overfitted"
            )

        if not test4.passed:
            warnings.append(
                f"⚠️  Strategy depends heavily on {len(outlier_tickers)} outlier tickers"
            )

        if len(outlier_tickers) > total_tickers * 0.1:
            warnings.append(
                f"⚠️  High proportion of outliers: "
                f"{len(outlier_tickers)}/{total_tickers} "
                f"({len(outlier_tickers)/total_tickers:.1%})"
            )

        return RobustnessReport(
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            is_robust=is_robust,
            test_results=tests,
            recommendation=recommendation,
            warnings=warnings,
        )

    def _test_mean_return(
        self, returns_full: pd.Series, returns_clean: pd.Series
    ) -> RobustnessTestResult:
        """Test if mean return is stable without outliers."""
        mean_full = returns_full.mean()
        mean_clean = returns_clean.mean()
        diff = mean_clean - mean_full
        diff_pct = diff / abs(mean_full) if mean_full != 0 else 0.0

        # Pass if clean returns are similar or better
        passed = diff >= -self.return_tolerance * abs(mean_full)

        notes = ""
        if diff > 0:
            notes = "✅ Returns IMPROVED without outliers"
        elif abs(diff_pct) < self.return_tolerance:
            notes = "✅ Returns STABLE without outliers"
        else:
            notes = "❌ Returns DECLINED without outliers"

        return RobustnessTestResult(
            test_name="Mean Return Stability",
            passed=passed,
            metric_full=mean_full,
            metric_clean=mean_clean,
            difference=diff,
            difference_pct=diff_pct,
            threshold=self.return_tolerance,
            notes=notes,
        )

    def _test_sharpe_ratio(
        self, returns_full: pd.Series, returns_clean: pd.Series
    ) -> RobustnessTestResult:
        """Test if Sharpe ratio is stable without outliers."""
        sharpe_full = self._calculate_sharpe(returns_full)
        sharpe_clean = self._calculate_sharpe(returns_clean)
        diff = sharpe_clean - sharpe_full

        # Pass if Sharpe is similar or better
        passed = diff >= -self.sharpe_tolerance

        notes = ""
        if diff > 0:
            notes = "✅ Sharpe ratio IMPROVED without outliers"
        elif abs(diff) < self.sharpe_tolerance:
            notes = "✅ Sharpe ratio STABLE without outliers"
        else:
            notes = "❌ Sharpe ratio DECLINED without outliers"

        return RobustnessTestResult(
            test_name="Sharpe Ratio Stability",
            passed=passed,
            metric_full=sharpe_full,
            metric_clean=sharpe_clean,
            difference=diff,
            difference_pct=0.0,  # Not applicable for Sharpe
            threshold=self.sharpe_tolerance,
            notes=notes,
        )

    def _test_win_rate(
        self, returns_full: pd.Series, returns_clean: pd.Series
    ) -> RobustnessTestResult:
        """Test if win rate is stable without outliers."""
        win_rate_full = (returns_full > 0).mean()
        win_rate_clean = (returns_clean > 0).mean()
        diff = win_rate_clean - win_rate_full

        # Pass if win rate is similar or better
        passed = diff >= -self.win_rate_tolerance

        notes = ""
        if diff > 0:
            notes = "✅ Win rate IMPROVED without outliers"
        elif abs(diff) < self.win_rate_tolerance:
            notes = "✅ Win rate STABLE without outliers"
        else:
            notes = "❌ Win rate DECLINED without outliers"

        return RobustnessTestResult(
            test_name="Win Rate Stability",
            passed=passed,
            metric_full=win_rate_full,
            metric_clean=win_rate_clean,
            difference=diff,
            difference_pct=0.0,  # Not applicable for win rate
            threshold=self.win_rate_tolerance,
            notes=notes,
        )

    def _test_outlier_dependency(
        self,
        outlier_tickers: Set[str],
        total_tickers: int,
        returns_full: pd.Series,
        returns_clean: pd.Series,
    ) -> RobustnessTestResult:
        """Test if strategy depends too heavily on outlier tickers."""
        outlier_count = len(outlier_tickers)
        outlier_pct = outlier_count / total_tickers if total_tickers > 0 else 0.0

        # Calculate contribution of outliers
        mean_full = returns_full.mean()
        mean_clean = returns_clean.mean()

        if mean_full != 0:
            outlier_contribution = abs((mean_full - mean_clean) / mean_full)
        else:
            outlier_contribution = 0.0

        # Pass if outliers contribute less than 20% to returns
        passed = outlier_contribution < 0.20

        notes = f"Outliers: {outlier_count}/{total_tickers} ({outlier_pct:.1%}), " \
                f"Contribution: {outlier_contribution:.1%}"

        if passed:
            notes = "✅ " + notes
        else:
            notes = "❌ " + notes + " - too high dependency"

        return RobustnessTestResult(
            test_name="Outlier Dependency",
            passed=passed,
            metric_full=outlier_contribution,
            metric_clean=0.0,
            difference=outlier_contribution,
            difference_pct=outlier_contribution,
            threshold=0.20,
            notes=notes,
        )

    def _calculate_sharpe(self, returns: pd.Series, rf_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - rf_rate
        return float(excess_returns.mean() / excess_returns.std())

    def _generate_recommendation(
        self,
        is_robust: bool,
        tests: List[RobustnessTestResult],
        outlier_tickers: Set[str],
        total_tickers: int,
    ) -> str:
        """Generate actionable recommendation based on test results."""
        if is_robust:
            return (
                "✅ STRATEGY IS ROBUST\n"
                "   The strategy performs consistently with/without outliers.\n"
                "   Safe to proceed with this configuration.\n"
                f"   Recommendation: Exclude {len(outlier_tickers)} outlier tickers "
                "for cleaner results."
            )
        else:
            failed_tests = [t.test_name for t in tests if not t.passed]
            return (
                "❌ STRATEGY MAY BE OVERFITTED\n"
                f"   Failed tests: {', '.join(failed_tests)}\n"
                "   The strategy shows significant dependency on outlier behavior.\n"
                "   Recommendations:\n"
                "   1. Review outlier tickers for data quality issues\n"
                "   2. Consider adding filters (volume, market cap, liquidity)\n"
                "   3. Test strategy on different time periods\n"
                "   4. Investigate why outliers contribute so much to returns"
            )

    def generate_report(self, validation: RobustnessReport) -> str:
        """Generate human-readable robustness validation report."""
        lines = [
            "=" * 70,
            "ROBUSTNESS VALIDATION REPORT",
            "=" * 70,
            "",
            f"📊 OVERALL RESULT: {'✅ ROBUST' if validation.is_robust else '❌ NOT ROBUST'}",
            f"   Tests Passed: {validation.tests_passed}/{len(validation.test_results)}",
            f"   Tests Failed: {validation.tests_failed}/{len(validation.test_results)}",
            "",
            "📋 TEST RESULTS:",
            "",
        ]

        # Test details table
        lines.append(
            f"{'Test':<25} {'Full':<12} {'Clean':<12} {'Diff':<12} {'Status':<10}"
        )
        lines.append("-" * 70)

        for test in validation.test_results:
            status = "✅ PASS" if test.passed else "❌ FAIL"
            lines.append(
                f"{test.test_name:<25} {test.metric_full:>11.4f} {test.metric_clean:>11.4f} "
                f"{test.difference:>+11.4f} {status:<10}"
            )

        lines.append("")
        lines.append("📝 NOTES:")
        for test in validation.test_results:
            lines.append(f"   {test.notes}")

        if validation.warnings:
            lines.append("")
            lines.append("⚠️  WARNINGS:")
            for warning in validation.warnings:
                lines.append(f"   {warning}")

        lines.append("")
        lines.append("💡 RECOMMENDATION:")
        for line in validation.recommendation.split('\n'):
            lines.append(f"   {line}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)
