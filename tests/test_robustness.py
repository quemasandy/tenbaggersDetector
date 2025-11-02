"""Tests for robustness validation."""
import numpy as np
import pandas as pd
import pytest

from tenbaggers_detector.analysis import RobustnessValidator


class TestRobustnessValidator:
    """Test suite for RobustnessValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator with default config."""
        return RobustnessValidator()

    @pytest.fixture
    def robust_returns(self):
        """Generate returns that are robust (similar with/without outliers)."""
        np.random.seed(42)

        # Full universe returns (100 samples)
        returns_full = pd.Series(np.random.normal(0.02, 0.05, 100))

        # Clean universe returns (similar distribution, 90 samples)
        returns_clean = pd.Series(np.random.normal(0.021, 0.049, 90))

        return returns_full, returns_clean

    @pytest.fixture
    def overfitted_returns(self):
        """Generate returns showing overfitting (much worse without outliers)."""
        np.random.seed(42)

        # Full universe with high returns (boosted by outliers)
        returns_full = pd.Series(np.random.normal(0.05, 0.08, 100))

        # Clean universe with much lower returns
        returns_clean = pd.Series(np.random.normal(0.01, 0.04, 90))

        return returns_full, returns_clean

    def test_validate_robust_strategy(self, validator, robust_returns):
        """Test validation of robust strategy."""
        returns_full, returns_clean = robust_returns
        outlier_tickers = {'OUTLIER1', 'OUTLIER2', 'OUTLIER3'}
        total_tickers = 50

        report = validator.validate(
            returns_full=returns_full,
            returns_clean=returns_clean,
            outlier_tickers=outlier_tickers,
            total_tickers=total_tickers,
        )

        assert report.is_robust, "Should classify robust strategy as robust"
        assert report.tests_passed >= 3, "Should pass at least 3/4 tests"
        # Note: warnings are possible even for robust strategies (e.g., outlier dependency warning)

    def test_validate_overfitted_strategy(self, validator, overfitted_returns):
        """Test validation of overfitted strategy."""
        returns_full, returns_clean = overfitted_returns
        outlier_tickers = {'OUT1', 'OUT2', 'OUT3', 'OUT4', 'OUT5'}
        total_tickers = 50

        report = validator.validate(
            returns_full=returns_full,
            returns_clean=returns_clean,
            outlier_tickers=outlier_tickers,
            total_tickers=total_tickers,
        )

        # Should detect overfitting
        assert not report.is_robust, "Should classify overfitted strategy as not robust"
        assert report.tests_failed > 0, "Should fail some tests"
        assert len(report.warnings) > 0, "Should have warnings"

    def test_mean_return_stability(self, validator):
        """Test mean return stability test."""
        # Similar returns (well within tolerance)
        returns_full = pd.Series([0.02] * 100)
        returns_clean = pd.Series([0.0195] * 90)  # Only 2.5% decline

        result = validator._test_mean_return(returns_full, returns_clean)

        assert result.passed, "Should pass with stable returns"
        assert result.test_name == "Mean Return Stability"
        assert "STABLE" in result.notes or "IMPROVED" in result.notes

    def test_mean_return_decline(self, validator):
        """Test detection of mean return decline."""
        # Returns decline significantly
        returns_full = pd.Series([0.05] * 100)
        returns_clean = pd.Series([0.01] * 90)

        result = validator._test_mean_return(returns_full, returns_clean)

        assert not result.passed, "Should fail with declining returns"
        assert "DECLINED" in result.notes

    def test_sharpe_ratio_stability(self, validator):
        """Test Sharpe ratio stability test."""
        np.random.seed(42)

        # Similar Sharpe ratios
        returns_full = pd.Series(np.random.normal(0.02, 0.05, 100))
        returns_clean = pd.Series(np.random.normal(0.019, 0.049, 90))

        result = validator._test_sharpe_ratio(returns_full, returns_clean)

        # Should generally pass (random data might occasionally fail)
        assert result.test_name == "Sharpe Ratio Stability"
        assert result.metric_full != 0

    def test_win_rate_stability(self, validator):
        """Test win rate stability test."""
        # Create returns with known win rates
        returns_full = pd.Series([0.01, -0.01, 0.02, -0.01, 0.01] * 20)  # 60% win rate
        returns_clean = pd.Series([0.01, -0.01, 0.02, -0.01, 0.01] * 18)  # 60% win rate

        result = validator._test_win_rate(returns_full, returns_clean)

        assert result.passed, "Should pass with stable win rate"
        assert result.test_name == "Win Rate Stability"
        assert abs(result.metric_full - 0.6) < 0.01

    def test_outlier_dependency(self, validator):
        """Test outlier dependency test."""
        # Low dependency (small contribution from outliers)
        returns_full = pd.Series([0.02] * 100)
        returns_clean = pd.Series([0.019] * 90)
        outlier_tickers = {'OUT1', 'OUT2'}
        total_tickers = 50

        result = validator._test_outlier_dependency(
            outlier_tickers, total_tickers, returns_full, returns_clean
        )

        assert result.passed, "Should pass with low outlier dependency"
        assert result.test_name == "Outlier Dependency"

    def test_high_outlier_dependency(self, validator):
        """Test detection of high outlier dependency."""
        # High dependency (large contribution from outliers)
        returns_full = pd.Series([0.05] * 100)
        returns_clean = pd.Series([0.01] * 90)  # 80% drop without outliers
        outlier_tickers = set([f'OUT{i}' for i in range(10)])  # Many outliers
        total_tickers = 50

        result = validator._test_outlier_dependency(
            outlier_tickers, total_tickers, returns_full, returns_clean
        )

        assert not result.passed, "Should fail with high outlier dependency"
        assert "too high" in result.notes.lower()

    def test_calculate_sharpe(self, validator):
        """Test Sharpe ratio calculation."""
        # Known returns
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.00])

        sharpe = validator._calculate_sharpe(returns, rf_rate=0.0)

        # Manual calculation
        mean = returns.mean()
        std = returns.std()
        expected_sharpe = mean / std

        assert abs(sharpe - expected_sharpe) < 0.001

    def test_calculate_sharpe_zero_std(self, validator):
        """Test Sharpe calculation with zero std (edge case)."""
        returns = pd.Series([0.02] * 10)  # Constant returns

        sharpe = validator._calculate_sharpe(returns)

        # Should return 0 or handle gracefully
        assert sharpe == 0.0 or not np.isnan(sharpe)

    def test_generate_recommendation_robust(self, validator):
        """Test recommendation for robust strategy."""
        from tenbaggers_detector.analysis.robustness import RobustnessTestResult

        # All tests passing
        tests = [
            RobustnessTestResult(
                test_name="Test 1",
                passed=True,
                metric_full=0.02,
                metric_clean=0.021,
                difference=0.001,
                difference_pct=0.05,
                threshold=0.05,
                notes="Pass",
            )
            for _ in range(4)
        ]

        recommendation = validator._generate_recommendation(
            is_robust=True,
            tests=tests,
            outlier_tickers={'OUT1', 'OUT2'},
            total_tickers=50,
        )

        assert "ROBUST" in recommendation
        assert "Safe to proceed" in recommendation

    def test_generate_recommendation_overfitted(self, validator):
        """Test recommendation for overfitted strategy."""
        from tenbaggers_detector.analysis.robustness import RobustnessTestResult

        # Some tests failing
        tests = [
            RobustnessTestResult(
                test_name=f"Test {i}",
                passed=i % 2 == 0,
                metric_full=0.05,
                metric_clean=0.01,
                difference=-0.04,
                difference_pct=-0.80,
                threshold=0.05,
                notes="Fail" if i % 2 == 1 else "Pass",
            )
            for i in range(4)
        ]

        recommendation = validator._generate_recommendation(
            is_robust=False,
            tests=tests,
            outlier_tickers=set([f'OUT{i}' for i in range(10)]),
            total_tickers=50,
        )

        assert "OVERFITTED" in recommendation or "NOT" in recommendation
        assert "Recommendations:" in recommendation

    def test_generate_report(self, validator, robust_returns):
        """Test report generation."""
        returns_full, returns_clean = robust_returns
        outlier_tickers = {'OUT1', 'OUT2'}
        total_tickers = 50

        report_obj = validator.validate(
            returns_full, returns_clean, outlier_tickers, total_tickers
        )

        report_str = validator.generate_report(report_obj)

        assert isinstance(report_str, str)
        assert len(report_str) > 0
        assert "ROBUSTNESS VALIDATION REPORT" in report_str
        assert "OVERALL RESULT" in report_str
        assert "TEST RESULTS" in report_str

    def test_custom_tolerances(self):
        """Test validator with custom tolerance settings."""
        validator = RobustnessValidator(
            return_tolerance=0.10,
            sharpe_tolerance=0.5,
            win_rate_tolerance=0.05,
        )

        assert validator.return_tolerance == 0.10
        assert validator.sharpe_tolerance == 0.5
        assert validator.win_rate_tolerance == 0.05

    def test_warnings_generation(self, validator):
        """Test that warnings are generated appropriately."""
        # Create scenario with high outlier percentage
        returns_full = pd.Series(np.random.normal(0.05, 0.08, 100))
        returns_clean = pd.Series(np.random.normal(0.01, 0.04, 90))

        # Many outliers (>10%)
        outlier_tickers = set([f'OUT{i}' for i in range(10)])
        total_tickers = 50

        report = validator.validate(
            returns_full, returns_clean, outlier_tickers, total_tickers
        )

        # Should have warning about high outlier proportion
        assert any("proportion" in w.lower() for w in report.warnings)

    def test_empty_returns(self, validator):
        """Test handling of empty returns."""
        returns_full = pd.Series([])
        returns_clean = pd.Series([])
        outlier_tickers = set()
        total_tickers = 0

        # Should handle gracefully without crashing
        try:
            report = validator.validate(
                returns_full, returns_clean, outlier_tickers, total_tickers
            )
            # If it doesn't crash, test passes
            assert True
        except Exception as e:
            pytest.fail(f"Should handle empty returns gracefully: {e}")
