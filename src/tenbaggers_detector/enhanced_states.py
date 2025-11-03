"""State machine implementation for the enhanced pipeline.

This module encapsulates the execution flow of :class:`EnhancedPipeline` in
explicit states, following the principles highlighted in the design theory:

* **Encapsulate the change** – each state owns the logic of a pipeline phase.
* **Polymorphism over conditionals** – transitions are handled via objects
  instead of nested ``if`` statements.
* **Single Responsibility** – the context orchestrates, states execute.
* **Open/Closed principle** – new phases can be added by implementing a new
  state without touching existing ones.

The state machine keeps the orchestration logic out of
``EnhancedPipeline.run`` so that strategies, validation layers, and future
extensions can evolve independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, TYPE_CHECKING

from .data.models import SignalResult

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from .enhanced_pipeline import EnhancedPipeline


@dataclass
class PipelineRunData:
    """Container with data shared across pipeline states."""

    tickers: List[str]
    start: str | None
    end: str | None
    initial_results: List[SignalResult] = field(default_factory=list)
    filtered_results: List[SignalResult] = field(default_factory=list)
    final_results: List[SignalResult] = field(default_factory=list)


class PipelineState(Protocol):
    """Protocol for pipeline states."""

    def handle(self, context: "EnhancedPipelineContext") -> Optional["PipelineState"]:
        """Execute the state and return the next state (or ``None`` to stop)."""


class EnhancedPipelineContext:
    """Context that manages state transitions for the enhanced pipeline."""

    def __init__(
        self,
        pipeline: "EnhancedPipeline",
        tickers: List[str],
        start: str | None,
        end: str | None,
    ) -> None:
        self.pipeline = pipeline
        self.data = PipelineRunData(tickers=tickers, start=start, end=end)

    def execute(self) -> None:
        """Run the state machine until completion."""
        state: Optional[PipelineState] = BaseDetectionState()
        while state is not None:
            state = state.handle(self)


class BaseDetectionState:
    """Run the base breakout detection."""

    def handle(self, context: EnhancedPipelineContext) -> Optional[PipelineState]:
        pipeline = context.pipeline
        pipeline._log("\n🔍 Step 1: Running base breakout detection...")

        results = pipeline._run_base_detection(
            context.data.tickers, context.data.start, context.data.end
        )

        if not results:
            pipeline._log("❌ No signals detected")
            context.data.final_results = []
            return None

        pipeline._log(f"✅ Found {len(results)} initial signals")
        context.data.initial_results = results
        context.data.filtered_results = list(results)
        return OutlierDetectionState()


class OutlierDetectionState:
    """Run statistical outlier detection when enabled."""

    def handle(self, context: EnhancedPipelineContext) -> Optional[PipelineState]:
        pipeline = context.pipeline

        if not pipeline.config.enable_outlier_detection:
            return SignalFilteringState()

        pipeline._log("\n🔍 Step 2: Running outlier detection...")
        pipeline._run_outlier_detection(context.data.initial_results)

        if pipeline.outlier_results:
            pipeline._log(pipeline.outlier_detector.generate_report(pipeline.outlier_results))

        return SignalFilteringState()


class SignalFilteringState:
    """Filter redundant signals and analyze quality."""

    def handle(self, context: EnhancedPipelineContext) -> Optional[PipelineState]:
        pipeline = context.pipeline
        initial_results = context.data.initial_results

        if pipeline.config.enable_signal_filtering:
            pipeline._log("\n🔍 Step 3: Filtering redundant signals...")
            filtered = pipeline._filter_redundant_signals(initial_results)
        else:
            pipeline._log("\nℹ️ Signal filtering disabled – skipping step.")
            filtered = list(initial_results)

        context.data.filtered_results = filtered
        pipeline._update_signal_quality(initial_results, filtered)
        return RobustnessValidationState()


class RobustnessValidationState:
    """Validate robustness when the configuration requires it."""

    def handle(self, context: EnhancedPipelineContext) -> Optional[PipelineState]:
        pipeline = context.pipeline

        if not pipeline.config.enable_robustness_validation:
            return CompletedState()

        if not pipeline.outlier_results:
            pipeline._log(
                "\nℹ️ Robustness validation skipped – requires prior outlier analysis."
            )
            return CompletedState()

        pipeline._log("\n🔍 Step 4: Validating robustness...")
        pipeline._run_robustness_validation(
            context.data.filtered_results, context.data.tickers
        )

        if pipeline.robustness_report:
            pipeline._log(pipeline.robustness_validator.generate_report(pipeline.robustness_report))

        return CompletedState()


class CompletedState:
    """Finalize the execution and publish the results."""

    def handle(self, context: EnhancedPipelineContext) -> Optional[PipelineState]:
        pipeline = context.pipeline
        final_results = context.data.filtered_results or context.data.initial_results
        context.data.final_results = final_results
        pipeline._log(f"\n✅ Final signals: {len(final_results)}")
        return None
