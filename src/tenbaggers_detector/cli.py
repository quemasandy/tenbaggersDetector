"""Command line interface for the tenbaggers detector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from .data.sources import CSVSource, MarketDataSource, YFinanceSource
from .pipeline import PipelineConfig, TenbaggerPipeline


def _build_source(args: argparse.Namespace) -> MarketDataSource:
    if args.source == "yfinance":
        return YFinanceSource(auto_adjust=not args.no_adjust)
    if args.source == "csv":
        if not args.csv_template:
            raise ValueError("--csv-template must be provided when using csv source")
        return CSVSource(path_template=args.csv_template)
    raise ValueError(f"Unknown source {args.source}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect early-stage tenbagger candidates")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols to evaluate")
    parser.add_argument("--start", help="Start date YYYY-MM-DD", default=None)
    parser.add_argument("--end", help="End date YYYY-MM-DD", default=None)
    parser.add_argument("--source", choices=["yfinance", "csv"], default="yfinance")
    parser.add_argument("--no-adjust", action="store_true", help="Disable adjusted prices when using yfinance")
    parser.add_argument("--csv-template", help="CSV path template e.g. data/{ticker}.csv")
    parser.add_argument("--min-price", type=float, default=40.0)
    parser.add_argument("--min-dollar-volume", type=float, default=1_000_000.0)
    parser.add_argument("--min-volume", type=float, default=300_000.0)
    parser.add_argument("--lookback-years", type=int, default=5)
    parser.add_argument("--bins", type=int, default=160)
    parser.add_argument("--zscore", type=float, default=2.0)
    parser.add_argument("--compression", type=float, default=0.35)
    parser.add_argument("--output", help="Output file for JSON results")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    source = _build_source(args)
    config = PipelineConfig()
    config.universe_filters.max_price = args.min_price
    config.universe_filters.min_dollar_volume = args.min_dollar_volume
    config.universe_filters.min_volume = args.min_volume
    config.breakout.lookback_days = args.lookback_years * 252
    config.breakout.volume_bins = args.bins
    config.breakout.zscore_threshold = args.zscore
    config.breakout.compression_percentile_max = args.compression

    pipeline = TenbaggerPipeline(source, config)
    results = pipeline.run(args.tickers, start=args.start, end=args.end)
    payload = [result.to_json() for result in results]
    text = json.dumps(payload, indent=2)

    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
