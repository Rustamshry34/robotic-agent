"""
Baseline experiment runner for the self-verifying robotic AI agent.

This module provides tools for running controlled ablation experiments
comparing different system configurations:

- baseline_a: LLM planning + robot tools only
- baseline_b: LLM planning + robot tools + experience memory
- baseline_c: LLM planning + robot tools + action verification
- full_system: LLM planning + robot tools + verification + memory + recovery

Each baseline can be run multiple times with different seeds to measure
variance and statistical significance.

Usage:
    # Python API
    experiment = BaselineExperiment(
        instruction="Take the red cup to the kitchen counter",
        baselines=["baseline_a", "baseline_b", "baseline_c", "full_system"],
        iterations=10,
    )
    result = await experiment.run()

    # CLI
    python -m evaluation.baselines \\
        --instruction "Take the red cup" \\
        --baselines baseline_a baseline_b baseline_c full_system \\
        --iterations 10 \\
        --output results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from app.agent.executor import (
    ExecutionResult,
    RobotExecutor,
    SimulatedMCPToolInvoker,
)
from app.agent.planner import create_planner
from app.core.config import Environment, Settings, get_settings
from mcp_server.tools import SimulatedRobotBackend
from memory.chroma_client import ChromaMemoryClient

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


logger = logging.getLogger(__name__)


__all__ = [
    "AVAILABLE_BASELINES",
    "BaselineConfig",
    "BaselineExperiment",
    "BaselineResult",
    "BaselineRunner",
    "ExperimentResult",
    "BaselineSummary",
    "run_baseline_experiment",
]


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------


AVAILABLE_BASELINES: tuple[str, ...] = (
    "baseline_a",
    "baseline_b",
    "baseline_c",
    "full_system",
)


# ----------------------------------------------------------------------
# Result schemas
# ----------------------------------------------------------------------


class BaselineResult(BaseModel):
    """
    Result of a single baseline run.

    Captures all metrics needed for academic comparison:
    - success/failure
    - execution time
    - step counts
    - recovery attempts
    - retry counts
    - verification outcomes
    - memory retrieval counts
    """

    baseline: str
    iteration: int
    seed: int

    success: bool
    status: str

    task_id: str | None = None
    task_name: str = ""
    instruction: str = ""

    error_code: str | None = None
    error_message: str | None = None

    executed_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    verified_steps: int = 0
    unverified_steps: int = 0

    recovery_attempts: int = 0
    retry_count: int = 0

    duration_ms: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None

    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return 1.0 if self.success else 0.0

    @property
    def verification_rate(self) -> float:
        if self.executed_steps == 0:
            return 0.0
        return self.verified_steps / self.executed_steps

    @property
    def recovery_rate(self) -> float:
        if not self.success:
            return 0.0
        if self.recovery_attempts == 0:
            return 0.0
        return 1.0


class BaselineSummary(BaseModel):
    """
    Aggregated statistics for a single baseline across multiple iterations.
    """

    baseline: str
    total_runs: int = 0

    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 0.0

    average_duration_ms: float = 0.0
    min_duration_ms: int = 0
    max_duration_ms: int = 0

    average_executed_steps: float = 0.0
    average_completed_steps: float = 0.0
    average_failed_steps: float = 0.0
    average_verified_steps: float = 0.0

    average_recovery_attempts: float = 0.0
    average_retry_count: float = 0.0
    total_recovery_attempts: int = 0
    total_retries: int = 0

    error_distribution: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def from_results(
        cls,
        baseline: str,
        results: list[BaselineResult],
    ) -> BaselineSummary:
        """
        Build summary statistics from a list of baseline results.
        """
        if not results:
            return cls(baseline=baseline)

        total = len(results)
        successes = sum(1 for r in results if r.success)
        failures = total - successes

        durations = [r.duration_ms for r in results]
        executed = [r.executed_steps for r in results]
        completed = [r.completed_steps for r in results]
        failed = [r.failed_steps for r in results]
        verified = [r.verified_steps for r in results]
        recoveries = [r.recovery_attempts for r in results]
        retries = [r.retry_count for r in results]

        error_distribution: dict[str, int] = {}
        for r in results:
            if r.error_code:
                error_distribution[r.error_code] = (
                    error_distribution.get(r.error_code, 0) + 1
                )

        return cls(
            baseline=baseline,
            total_runs=total,
            success_count=successes,
            failure_count=failures,
            success_rate=successes / total if total > 0 else 0.0,
            average_duration_ms=sum(durations) / total,
            min_duration_ms=min(durations) if durations else 0,
            max_duration_ms=max(durations) if durations else 0,
            average_executed_steps=sum(executed) / total,
            average_completed_steps=sum(completed) / total,
            average_failed_steps=sum(failed) / total,
            average_verified_steps=sum(verified) / total,
            average_recovery_attempts=sum(recoveries) / total,
            average_retry_count=sum(retries) / total,
            total_recovery_attempts=sum(recoveries),
            total_retries=sum(retries),
            error_distribution=error_distribution,
        )


class ExperimentResult(BaseModel):
    """
    Complete result of a baseline comparison experiment.
    """

    experiment_id: str
    instruction: str
    baselines: list[str]
    iterations: int
    seed: int

    started_at: datetime
    finished_at: datetime | None = None
    total_duration_ms: int = 0

    results: list[BaselineResult] = Field(default_factory=list)
    summaries: dict[str, BaselineSummary] = Field(default_factory=dict)

    metadata: dict[str, Any] = Field(default_factory=dict)

    def compute_summaries(self) -> None:
        """
        Compute summary statistics for each baseline.
        """
        for baseline in self.baselines:
            baseline_results = [
                r for r in self.results if r.baseline == baseline
            ]
            self.summaries[baseline] = BaselineSummary.from_results(
                baseline=baseline,
                results=baseline_results,
            )


# ----------------------------------------------------------------------
# Baseline configuration
# ----------------------------------------------------------------------


class BaselineConfig(BaseModel):
    """
    Configuration for a single baseline run.
    """

    baseline: str
    enable_memory: bool = False
    enable_verification: bool = False
    enable_recovery: bool = False

    @classmethod
    def for_baseline(cls, baseline: str) -> BaselineConfig:
        """
        Create config for a named baseline.
        """
        baseline = baseline.strip().lower()

        if baseline == "baseline_a":
            return cls(
                baseline=baseline,
                enable_memory=False,
                enable_verification=False,
                enable_recovery=False,
            )

        if baseline == "baseline_b":
            return cls(
                baseline=baseline,
                enable_memory=True,
                enable_verification=False,
                enable_recovery=False,
            )

        if baseline == "baseline_c":
            return cls(
                baseline=baseline,
                enable_memory=False,
                enable_verification=True,
                enable_recovery=False,
            )

        if baseline == "full_system":
            return cls(
                baseline=baseline,
                enable_memory=True,
                enable_verification=True,
                enable_recovery=True,
            )

        raise ValueError(
            f"Unknown baseline: '{baseline}'. "
            f"Available: {AVAILABLE_BASELINES}"
        )


# ----------------------------------------------------------------------
# Baseline runner
# ----------------------------------------------------------------------


class BaselineRunner:
    """
    Runs a single baseline configuration.

    Responsible for:
    - Creating planner with correct baseline settings
    - Creating executor with correct config
    - Running task execution
    - Collecting metrics
    - Returning structured result
    """

    def __init__(
        self,
        config: BaselineConfig,
        settings: Settings | None = None,
        memory_client: ChromaMemoryClient | None = None,
    ) -> None:
        self.config = config
        self.settings = settings or get_settings()

        if memory_client is None and config.enable_memory:
            try:
                self.memory_client = ChromaMemoryClient.from_env()
            except Exception:
                logger.exception(
                    "Could not initialize ChromaMemoryClient. "
                    "Memory will be disabled for this baseline."
                )
                self.memory_client = None
        else:
            self.memory_client = memory_client

    def _create_planner(self) -> Any:
        """
        Create planner configured for this baseline.
        """
        return create_planner(
            settings=self.settings,
            memory_client=(
                self.memory_client
                if self.config.enable_memory
                else None
            ),
            baseline=self.config.baseline,
        )

    def _create_executor(self) -> RobotExecutor:
        """
        Create executor configured for this baseline.
        """
        planner = self._create_planner()

        return RobotExecutor(
            planner=planner,
            settings=self.settings,
            memory_client=(
                self.memory_client
                if self.config.enable_memory
                else None
            ),
        )

    async def run(
        self,
        instruction: str,
        iteration: int = 1,
        seed: int = 0,
        task_name: str | None = None,
        db_session: AsyncSession | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BaselineResult:
        """
        Run a single baseline iteration.
        """
        random.seed(seed)

        executor = self._create_executor()
        backend = SimulatedRobotBackend()
        tool_invoker = SimulatedMCPToolInvoker(backend=backend)

        started_at = datetime.now(UTC)

        try:
            execution_result: ExecutionResult = await executor.execute_task(
                instruction=instruction,
                tool_invoker=tool_invoker,
                task_name=task_name,
                baseline=self.config.baseline,
                db_session=db_session,
                metadata={
                    **(metadata or {}),
                    "baseline": self.config.baseline,
                    "iteration": iteration,
                    "seed": seed,
                },
            )
        except Exception as exc:
            logger.exception(
                "Baseline run failed: baseline=%s iteration=%d",
                self.config.baseline,
                iteration,
            )

            finished_at = datetime.now(UTC)

            return BaselineResult(
                baseline=self.config.baseline,
                iteration=iteration,
                seed=seed,
                success=False,
                status="error",
                instruction=instruction,
                error_code="BASELINE_EXCEPTION",
                error_message=str(exc),
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=int(
                    (finished_at - started_at).total_seconds() * 1000
                ),
            )

        finished_at = datetime.now(UTC)

        verified_steps = sum(
            1 for step in execution_result.steps if step.verified
        )
        unverified_steps = len(execution_result.steps) - verified_steps

        return BaselineResult(
            baseline=self.config.baseline,
            iteration=iteration,
            seed=seed,
            success=execution_result.success,
            status=execution_result.status,
            task_id=execution_result.task_id,
            task_name=execution_result.task_name,
            instruction=instruction,
            error_code=execution_result.error_code,
            error_message=execution_result.error_message,
            executed_steps=len(execution_result.steps),
            completed_steps=sum(
                1 for step in execution_result.steps if step.success
            ),
            failed_steps=sum(
                1 for step in execution_result.steps if not step.success
            ),
            verified_steps=verified_steps,
            unverified_steps=unverified_steps,
            recovery_attempts=execution_result.recovery_attempts,
            retry_count=execution_result.retry_count,
            duration_ms=execution_result.duration_ms,
            started_at=started_at,
            finished_at=finished_at,
            metrics=execution_result.metrics,
            metadata={
                **(metadata or {}),
                "baseline": self.config.baseline,
                "iteration": iteration,
                "seed": seed,
                "enable_memory": self.config.enable_memory,
                "enable_verification": self.config.enable_verification,
                "enable_recovery": self.config.enable_recovery,
            },
        )


# ----------------------------------------------------------------------
# Baseline experiment
# ----------------------------------------------------------------------


class BaselineExperiment:
    """
    Runs a complete baseline comparison experiment.

    Executes the same instruction across multiple baselines
    and multiple iterations, collecting metrics for comparison.
    """

    def __init__(
        self,
        instruction: str,
        baselines: list[str] | None = None,
        iterations: int = 1,
        settings: Settings | None = None,
        memory_client: ChromaMemoryClient | None = None,
        seed: int | None = None,
        task_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.instruction = instruction
        self.baselines = baselines or list(AVAILABLE_BASELINES)
        self.iterations = iterations
        self.settings = settings or get_settings()
        self.memory_client = memory_client
        self.seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        self.task_name = task_name
        self.metadata = metadata or {}

        self._validate_baselines()

    def _validate_baselines(self) -> None:
        """
        Validate that all baselines are known.
        """
        for baseline in self.baselines:
            if baseline not in AVAILABLE_BASELINES:
                raise ValueError(
                    f"Unknown baseline: '{baseline}'. "
                    f"Available: {AVAILABLE_BASELINES}"
                )

    async def run(
        self,
        db_session: AsyncSession | None = None,
    ) -> ExperimentResult:
        """
        Run the complete experiment.
        """
        experiment_id = str(uuid.uuid4())
        started_at = datetime.now(UTC)

        logger.info(
            "Starting baseline experiment: id=%s instruction=%r "
            "baselines=%s iterations=%d seed=%d",
            experiment_id,
            self.instruction,
            self.baselines,
            self.iterations,
            self.seed,
        )

        experiment_result = ExperimentResult(
            experiment_id=experiment_id,
            instruction=self.instruction,
            baselines=list(self.baselines),
            iterations=self.iterations,
            seed=self.seed,
            started_at=started_at,
            metadata=self.metadata,
        )

        for baseline in self.baselines:
            config = BaselineConfig.for_baseline(baseline)
            runner = BaselineRunner(
                config=config,
                settings=self.settings,
                memory_client=self.memory_client,
            )

            for iteration in range(1, self.iterations + 1):
                iteration_seed = self.seed + hash(
                    (baseline, iteration)
                ) % (2**31)

                logger.info(
                    "Running baseline=%s iteration=%d/%d seed=%d",
                    baseline,
                    iteration,
                    self.iterations,
                    iteration_seed,
                )

                result = await runner.run(
                    instruction=self.instruction,
                    iteration=iteration,
                    seed=iteration_seed,
                    task_name=self.task_name,
                    db_session=db_session,
                    metadata={
                        **self.metadata,
                        "experiment_id": experiment_id,
                    },
                )

                experiment_result.results.append(result)

        experiment_result.compute_summaries()

        finished_at = datetime.now(UTC)
        experiment_result.finished_at = finished_at
        experiment_result.total_duration_ms = int(
            (finished_at - started_at).total_seconds() * 1000
        )

        logger.info(
            "Baseline experiment completed: id=%s duration_ms=%d",
            experiment_id,
            experiment_result.total_duration_ms,
        )

        return experiment_result

    def print_summary(self, result: ExperimentResult) -> None:
        """
        Print a human-readable summary of the experiment results.
        """
        print("\n" + "=" * 80)
        print(f"BASELINE EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"Experiment ID: {result.experiment_id}")
        print(f"Instruction: {result.instruction}")
        print(f"Baselines: {', '.join(result.baselines)}")
        print(f"Iterations: {result.iterations}")
        print(f"Seed: {result.seed}")
        print(f"Total Duration: {result.total_duration_ms} ms")
        print("=" * 80)

        for baseline in result.baselines:
            summary = result.summaries.get(baseline)
            if summary is None:
                continue

            print(f"\n[{baseline.upper()}]")
            print(f"  Total Runs:              {summary.total_runs}")
            print(f"  Success Rate:            {summary.success_rate:.1%}")
            print(f"  Avg Duration:            {summary.average_duration_ms:.1f} ms")
            print(f"  Duration Range:          [{summary.min_duration_ms}, {summary.max_duration_ms}] ms")
            print(f"  Avg Executed Steps:      {summary.average_executed_steps:.2f}")
            print(f"  Avg Completed Steps:     {summary.average_completed_steps:.2f}")
            print(f"  Avg Failed Steps:        {summary.average_failed_steps:.2f}")
            print(f"  Avg Verified Steps:      {summary.average_verified_steps:.2f}")
            print(f"  Avg Recovery Attempts:   {summary.average_recovery_attempts:.2f}")
            print(f"  Avg Retries:             {summary.average_retry_count:.2f}")
            print(f"  Total Recoveries:        {summary.total_recovery_attempts}")
            print(f"  Total Retries:           {summary.total_retries}")

            if summary.error_distribution:
                print(f"  Error Distribution:")
                for error_code, count in sorted(
                    summary.error_distribution.items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    print(f"    - {error_code}: {count}")

        print("\n" + "=" * 80)


# ----------------------------------------------------------------------
# Convenience function
# ----------------------------------------------------------------------


async def run_baseline_experiment(
    instruction: str,
    baselines: list[str] | None = None,
    iterations: int = 1,
    settings: Settings | None = None,
    seed: int | None = None,
    output_path: str | None = None,
    print_summary: bool = True,
) -> ExperimentResult:
    """
    Run a baseline experiment and optionally save results to JSON.

    Args:
        instruction: Natural-language task instruction.
        baselines: List of baselines to compare.
        iterations: Number of iterations per baseline.
        settings: Optional app settings.
        seed: Optional random seed for reproducibility.
        output_path: Optional path to save JSON results.
        print_summary: Whether to print human-readable summary.

    Returns:
        ExperimentResult with all metrics.
    """
    experiment = BaselineExperiment(
        instruction=instruction,
        baselines=baselines,
        iterations=iterations,
        settings=settings,
        seed=seed,
    )

    result = await experiment.run()

    if print_summary:
        experiment.print_summary(result)

    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(
                result.model_dump(mode="json"),
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info("Results saved to: %s", output_file)

    return result


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run baseline comparison experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all baselines with 10 iterations
    python -m evaluation.baselines \\
        --instruction "Take the red cup to the kitchen counter" \\
        --iterations 10

    # Run specific baselines
    python -m evaluation.baselines \\
        --instruction "Take the red cup" \\
        --baselines baseline_a baseline_c full_system \\
        --iterations 5

    # Save results to JSON
    python -m evaluation.baselines \\
        --instruction "Take the red cup" \\
        --iterations 10 \\
        --output results.json
        """,
    )

    parser.add_argument(
        "--instruction",
        required=True,
        help="Natural-language task instruction.",
    )

    parser.add_argument(
        "--baselines",
        nargs="+",
        default=list(AVAILABLE_BASELINES),
        choices=AVAILABLE_BASELINES,
        help="Baselines to compare (default: all).",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations per baseline (default: 1).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress summary output.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )

    return parser.parse_args()


def main() -> int:
    """
    CLI entry point.
    """
    args = parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    try:
        asyncio.run(
            run_baseline_experiment(
                instruction=args.instruction,
                baselines=args.baselines,
                iterations=args.iterations,
                seed=args.seed,
                output_path=args.output,
                print_summary=not args.quiet,
            )
        )
        return 0

    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user.")
        return 130

    except Exception:
        logger.exception("Experiment failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
