"""Tenbaggers detector package."""
from .cli import main
from .pipeline import PipelineConfig, TenbaggerPipeline
from .signals.breakout import BreakoutConfig, BreakoutDetector

__all__ = [
    "main",
    "PipelineConfig",
    "TenbaggerPipeline",
    "BreakoutConfig",
    "BreakoutDetector",
]
