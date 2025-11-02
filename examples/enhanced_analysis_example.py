"""Example: Enhanced analysis with outlier detection and robustness validation.

This example demonstrates how to use the enhanced pipeline to:
1. Detect breakout signals
2. Identify statistical outliers
3. Filter redundant signals
4. Validate strategy robustness
5. Generate comprehensive reports

Run with:
    python examples/enhanced_analysis_example.py
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tenbaggers_detector.analysis import (
    OutlierConfig,
    OutlierDetector,
    RobustnessValidator,
    SignalQualityAnalyzer,
)
from tenbaggers_detector.data.sources import YFinanceSource
from tenbaggers_detector.enhanced_pipeline import EnhancedConfig, EnhancedPipeline


def main():
    """Run enhanced analysis example."""
    print("=" * 70)
    print("ENHANCED TENBAGGERS DETECTOR - OUTLIER ANALYSIS EXAMPLE")
    print("=" * 70)
    print()

    # =========================================================================
    # STEP 1: Configure enhanced pipeline
    # =========================================================================
    print("📋 Step 1: Configuring enhanced pipeline...")

    # Custom outlier detection config
    outlier_config = OutlierConfig(
        ks_threshold_severe=0.40,     # Auto-exclude if KS > 0.40
        ks_threshold_moderate=0.30,   # Flag for review if KS > 0.30
        pvalue_threshold=0.05,         # Statistical significance level
        min_signals=10,                # Minimum signals for analysis
    )

    # Enhanced pipeline config
    config = EnhancedConfig(
        # Outlier detection
        enable_outlier_detection=True,
        outlier_config=outlier_config,
        exclude_outliers=True,

        # Signal filtering
        enable_signal_filtering=True,
        filter_lookahead_days=5,

        # Robustness validation
        enable_robustness_validation=True,
        return_tolerance=0.05,
        sharpe_tolerance=0.30,

        # Reporting
        verbose=True,
    )

    print("✅ Configuration complete")
    print()

    # =========================================================================
    # STEP 2: Initialize pipeline
    # =========================================================================
    print("📋 Step 2: Initializing pipeline...")

    source = YFinanceSource()
    pipeline = EnhancedPipeline(source, config)

    print("✅ Pipeline initialized")
    print()

    # =========================================================================
    # STEP 3: Define ticker universe
    # =========================================================================
    print("📋 Step 3: Defining ticker universe...")

    # Example ticker list (mix of normal and potentially problematic tickers)
    tickers = [
        # Large cap tech (likely normal)
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',

        # High growth tech
        'NVDA', 'TSLA', 'AMD', 'CRM', 'ADBE',

        # Other sectors
        'JPM', 'BAC', 'WMT', 'PG', 'JNJ',
        'XOM', 'CVX', 'UNH', 'V', 'MA',
    ]

    print(f"✅ Analyzing {len(tickers)} tickers")
    print()

    # =========================================================================
    # STEP 4: Run enhanced analysis
    # =========================================================================
    print("📋 Step 4: Running enhanced analysis...")
    print("This may take a few minutes...")
    print()

    try:
        results = pipeline.run(
            tickers,
            start='2020-01-01',
            end='2023-12-31',
        )

        print()
        print(f"✅ Analysis complete - found {len(results)} signals")
        print()

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # STEP 5: Display results
    # =========================================================================
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    if results:
        print(f"📊 SIGNALS DETECTED: {len(results)}")
        print()
        print("Top 5 signals by score:")
        print()

        # Sort by score
        results_sorted = sorted(results, key=lambda x: x.score, reverse=True)

        for i, result in enumerate(results_sorted[:5], 1):
            print(f"{i}. {result.ticker}")
            print(f"   Date: {result.date_signal.strftime('%Y-%m-%d')}")
            print(f"   Score: {result.score:.1f}/100")
            print(f"   Price: ${result.price_close:.2f}")
            print(f"   Breakout Level: ${result.breakout_level:.2f}")
            print(f"   Volume Z-Score: {result.z_volume:.2f}")
            print(f"   Notes: {result.notes}")
            print()
    else:
        print("❌ No signals detected")
        print()

    # =========================================================================
    # STEP 6: Display analysis reports
    # =========================================================================
    print("=" * 70)
    print("STATISTICAL ANALYSIS REPORTS")
    print("=" * 70)
    print()

    # Full analysis report
    full_report = pipeline.get_analysis_report()
    print(full_report)
    print()

    # =========================================================================
    # STEP 7: Summary and recommendations
    # =========================================================================
    print("=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    print()

    # Outliers summary
    if pipeline.excluded_tickers:
        print(f"⚠️  EXCLUDED TICKERS ({len(pipeline.excluded_tickers)}):")
        for ticker in sorted(pipeline.excluded_tickers):
            print(f"   - {ticker}")
        print()

    # Robustness summary
    if pipeline.robustness_report:
        report = pipeline.robustness_report

        if report.is_robust:
            print("✅ STRATEGY IS ROBUST")
            print(f"   Passed {report.tests_passed}/{len(report.test_results)} tests")
            print("   Safe to proceed with this configuration")
        else:
            print("⚠️  STRATEGY MAY BE OVERFITTED")
            print(f"   Failed {report.tests_failed}/{len(report.test_results)} tests")
            print("   Review recommendations before proceeding")

        if report.warnings:
            print()
            print("⚠️  WARNINGS:")
            for warning in report.warnings:
                print(f"   {warning}")

    print()

    # =========================================================================
    # STEP 8: Export results (optional)
    # =========================================================================
    if results and input("Export results to JSON? (y/n): ").lower() == 'y':
        import json
        from datetime import datetime

        output_file = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert results to dict
        results_dict = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_tickers_analyzed': len(tickers),
                'total_signals': len(results),
                'outliers_excluded': len(pipeline.excluded_tickers),
                'is_robust': pipeline.robustness_report.is_robust if pipeline.robustness_report else None,
            },
            'signals': [
                {
                    'ticker': r.ticker,
                    'date': r.date_signal.isoformat(),
                    'score': r.score,
                    'price_close': r.price_close,
                    'breakout_level': r.breakout_level,
                    'z_volume': r.z_volume,
                    'notes': r.notes,
                }
                for r in results_sorted
            ],
            'excluded_tickers': list(pipeline.excluded_tickers),
        }

        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"✅ Results exported to {output_file}")

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
