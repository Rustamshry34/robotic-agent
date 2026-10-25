"""
Advanced metrics, statistical analysis, and report generation for baseline experiments.

This module provides:
- Metric calculation (success rate, step efficiency, recovery rate, etc.)
- Statistical significance testing (Bootstrap 95% Confidence Intervals)
- Report generation (Markdown for academic papers, CSV for data analysis)

Dependencies:
    numpy (for bootstrap resampling)
    statistics (for basic descriptive stats)
"""

from __future__ import annotations

import csv
import logging
import math
import random
import statistics
from pathlib import Path
from typing import Any, Optional

import numpy as np

from evaluation.baselines import (
    BaselineResult,
    BaselineSummary,
    ExperimentResult,
)


logger = logging.getLogger(__name__)


__all__ = [
    "MetricCalculator",
    "ReportGenerator",
    "StatisticalTester",
    "compute_bootstrap_ci",
]


# ----------------------------------------------------------------------
# Statistical Helpers
# ----------------------------------------------------------------------


def compute_bootstrap_ci(
    data: list[float],
    statistic: callable = np.mean,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a given statistic.

    Args:
        data: List of numeric values.
        statistic: Function to compute the statistic (e.g., np.mean, np.median).
        n_bootstraps: Number of bootstrap resamples.
        confidence_level: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (lower_bound, statistic_value, upper_bound).
    """
    if not data or len(data) < 2:
        val = statistic(data) if data else 0.0
        return val, val, val

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    data_array = np.array(data)
    n = len(data_array)
    
    # Generate bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstraps):
        sample = np.random.choice(data_array, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    bootstrap_stats = np.array(bootstrap_stats)
    
    # Compute percentiles
    alpha = 1.0 - confidence_level
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    center = statistic(data_array)

    return float(lower), float(center), float(upper)


# ----------------------------------------------------------------------
# Metric Calculator
# ----------------------------------------------------------------------


class MetricCalculator:
    """
    Computes advanced metrics from experiment results.
    """

    @staticmethod
    def calculate_step_efficiency(summary: BaselineSummary) -> float:
        """
        Calculate step efficiency: completed_steps / executed_steps.
        Higher is better (fewer wasted or failed steps).
        """
        if summary.average_executed_steps == 0:
            return 0.0
        return summary.average_completed_steps / summary.average_executed_steps

    @staticmethod
    def calculate_recovery_success_rate(summary: BaselineSummary) -> float:
        """
        Calculate recovery success rate.
        Defined as: (Successes that had recovery attempts) / (Total recovery attempts)
        Note: This is an approximation. A more precise version requires per-run data.
        """
        if summary.total_recovery_attempts == 0:
            return 0.0
        
        # Heuristic: If success rate is high and recovery attempts exist, 
        # many recoveries were likely successful. 
        # For precise calculation, we'd iterate over BaselineResult list.
        return min(1.0, summary.success_rate)  # Simplified proxy

    @classmethod
    def enrich_summary(cls, summary: BaselineSummary) -> dict[str, Any]:
        """
        Add derived metrics to a BaselineSummary.
        """
        enriched = summary.model_dump()
        enriched["step_efficiency"] = cls.calculate_step_efficiency(summary)
        enriched["recovery_success_rate_proxy"] = cls.calculate_recovery_success_rate(summary)
        return enriched


# ----------------------------------------------------------------------
# Statistical Tester
# ----------------------------------------------------------------------


class StatisticalTester:
    """
    Performs statistical comparisons between baselines.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def compare_to_baseline(
        self,
        target_results: list[BaselineResult],
        reference_results: list[BaselineResult],
        metric: str = "duration_ms",
    ) -> dict[str, Any]:
        """
        Compare a target baseline against a reference baseline (e.g., full_system vs baseline_a).

        Args:
            target_results: Results from the target baseline (e.g., full_system).
            reference_results: Results from the reference baseline (e.g., baseline_a).
            metric: The metric to compare ('duration_ms', 'success_rate', etc.).

        Returns:
            Dictionary with mean differences and 95% confidence intervals.
        """
        if metric == "success_rate":
            target_data = [1.0 if r.success else 0.0 for r in target_results]
            ref_data = [1.0 if r.success else 0.0 for r in reference_results]
            stat_func = np.mean
        elif metric == "duration_ms":
            target_data = [r.duration_ms for r in target_results]
            ref_data = [r.duration_ms for r in reference_results]
            stat_func = np.mean
        else:
            raise ValueError(f"Unsupported metric for comparison: {metric}")

        # Compute individual CIs
        target_lower, target_mean, target_upper = compute_bootstrap_ci(
            target_data, statistic=stat_func, seed=self.seed
        )
        ref_lower, ref_mean, ref_upper = compute_bootstrap_ci(
            ref_data, statistic=stat_func, seed=self.seed
        )

        # Compute difference CI (target - reference)
        # For difference, we bootstrap the paired/unpaired differences
        diff_data = [t - r for t, r in zip(target_data, ref_data)]
        # If lengths differ, truncate to min length for paired comparison, or use independent bootstrap
        if len(target_data) != len(ref_data):
            min_len = min(len(target_data), len(ref_data))
            diff_data = [target_data[i] - ref_data[i] for i in range(min_len)]
            
        diff_lower, diff_mean, diff_upper = compute_bootstrap_ci(
            diff_data, statistic=np.mean, seed=self.seed
        )

        # Determine significance (if 0 is not in the CI of the difference)
        is_significant = (diff_lower > 0) or (diff_upper < 0)

        return {
            "metric": metric,
            "target_mean": target_mean,
            "target_ci_95": (target_lower, target_upper),
            "reference_mean": ref_mean,
            "reference_ci_95": (ref_lower, ref_upper),
            "difference_mean": diff_mean,
            "difference_ci_95": (diff_lower, diff_upper),
            "is_significant": is_significant,
            "p_value_proxy": "< 0.05" if is_significant else ">= 0.05",
        }


# ----------------------------------------------------------------------
# Report Generator
# ----------------------------------------------------------------------


class ReportGenerator:
    """
    Generates human-readable and machine-readable reports from experiment results.
    """

    def __init__(self, result: ExperimentResult):
        self.result = result
        self.tester = StatisticalTester(seed=result.seed)

    def generate_markdown(self, output_path: str | Path) -> None:
        """
        Generate a Markdown report suitable for academic papers or READMEs.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        lines.append("# Baseline Experiment Report\n")
        lines.append(f"- **Experiment ID**: `{self.result.experiment_id}`")
        lines.append(f"- **Instruction**: {self.result.instruction}")
        lines.append(f"- **Iterations per Baseline**: {self.result.iterations}")
        lines.append(f"- **Seed**: {self.result.seed}")
        lines.append(f"- **Total Duration**: {self.result.total_duration_ms / 1000:.2f} seconds\n")

        lines.append("## 1. Summary Statistics\n")
        lines.append("| Baseline | Success Rate | Avg Duration (ms) | Avg Steps | Recovery Attempts |")
        lines.append("|----------|--------------|-------------------|-----------|-------------------|")

        for baseline in self.result.baselines:
            summary = self.result.summaries.get(baseline)
            if not summary:
                continue

            enriched = MetricCalculator.enrich_summary(summary)
            lines.append(
                f"| {baseline} | "
                f"{enriched['success_rate']:.1%} | "
                f"{enriched['average_duration_ms']:.1f} ± {self._ci_width(enriched['average_duration_ms'], summary.min_duration_ms, summary.max_duration_ms):.1f} | "
                f"{enriched['average_executed_steps']:.2f} | "
                f"{enriched['average_recovery_attempts']:.2f} |"
            )

        lines.append("\n## 2. Statistical Significance (vs `baseline_a`)\n")
        lines.append("Comparing key metrics against the simplest baseline (`baseline_a`) using 95% Bootstrap Confidence Intervals.\n")

        ref_baseline = "baseline_a"
        ref_results = [r for r in self.result.results if r.baseline == ref_baseline]

        for baseline in self.result.baselines:
            if baseline == ref_baseline:
                continue

            target_results = [r for r in self.result.results if r.baseline == baseline]
            if not target_results or not ref_results:
                continue

            lines.append(f"### {baseline} vs {ref_baseline}")
            
            # Duration comparison
            dur_stats = self.tester.compare_to_results(target_results, ref_results, "duration_ms")
            sig_marker = "✅ Significant" if dur_stats["is_significant"] else "❌ Not Significant"
            lines.append(f"- **Duration**: Mean diff = {dur_stats['difference_mean']:.1f} ms (95% CI: [{dur_stats['difference_ci_95'][0]:.1f}, {dur_stats['difference_ci_95'][1]:.1f}]) → {sig_marker}")

            # Success rate comparison
            succ_stats = self.tester.compare_to_results(target_results, ref_results, "success_rate")
            sig_marker = "✅ Significant" if succ_stats["is_significant"] else "❌ Not Significant"
            lines.append(f"- **Success Rate**: Mean diff = {succ_stats['difference_mean']:.1%} (95% CI: [{succ_stats['difference_ci_95'][0]:.1%}, {succ_stats['difference_ci_95'][1]:.1%}]) → {sig_marker}")
            lines.append("")

        lines.append("## 3. Error Distribution\n")
        for baseline in self.result.baselines:
            summary = self.result.summaries.get(baseline)
            if not summary or not summary.error_distribution:
                continue
            
            lines.append(f"### {baseline}")
            for err, count in sorted(summary.error_distribution.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"- `{err}`: {count} occurrences")
            lines.append("")

        with path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info("Markdown report saved to: %s", path)

    def generate_csv(self, output_path: str | Path) -> None:
        """
        Generate a CSV file with raw per-iteration results for external analysis (e.g., Python/R).
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "experiment_id", "baseline", "iteration", "seed",
            "success", "status", "duration_ms",
            "executed_steps", "completed_steps", "failed_steps", "verified_steps",
            "recovery_attempts", "retry_count", "error_code"
        ]

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in self.result.results:
                writer.writerow({
                    "experiment_id": self.result.experiment_id,
                    "baseline": r.baseline,
                    "iteration": r.iteration,
                    "seed": r.seed,
                    "success": r.success,
                    "status": r.status,
                    "duration_ms": r.duration_ms,
                    "executed_steps": r.executed_steps,
                    "completed_steps": r.completed_steps,
                    "failed_steps": r.failed_steps,
                    "verified_steps": r.verified_steps,
                    "recovery_attempts": r.recovery_attempts,
                    "retry_count": r.retry_count,
                    "error_code": r.error_code or "",
                })

        logger.info("CSV report saved to: %s", path)

    def _ci_width(self, mean: float, min_val: float, max_val: float) -> float:
        """
        Rough proxy for CI width when full bootstrap isn't pre-calculated in summary.
        """
        return (max_val - min_val) / 4.0  # Rule of thumb for small N


# ----------------------------------------------------------------------
# Convenience wrapper for StatisticalTester to use list of results
# ----------------------------------------------------------------------

def _patch_compare_to_results(self, target_results: list[BaselineResult], ref_results: list[BaselineResult], metric: str) -> dict[str, Any]:
    """
    Helper to bind compare_to_results to StatisticalTester dynamically or use as standalone.
    """
    if metric == "success_rate":
        target_data = [1.0 if r.success else 0.0 for r in target_results]
        ref_data = [1.0 if r.success else 0.0 for r in ref_results]
        stat_func = np.mean
    elif metric == "duration_ms":
        target_data = [r.duration_ms for r in target_results]
        ref_data = [r.duration_ms for r in ref_results]
        stat_func = np.mean
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    target_lower, target_mean, target_upper = compute_bootstrap_ci(target_data, statistic=stat_func, seed=self.seed)
    ref_lower, ref_mean, ref_upper = compute_bootstrap_ci(ref_data, statistic=stat_func, seed=self.seed)

    min_len = min(len(target_data), len(ref_data))
    diff_data = [target_data[i] - ref_data[i] for i in range(min_len)]
    diff_lower, diff_mean, diff_upper = compute_bootstrap_ci(diff_data, statistic=np.mean, seed=self.seed)

    is_significant = (diff_lower > 0) or (diff_upper < 0)

    return {
        "metric": metric,
        "target_mean": target_mean,
        "target_ci_95": (target_lower, target_upper),
        "reference_mean": ref_mean,
        "reference_ci_95": (ref_lower, ref_upper),
        "difference_mean": diff_mean,
        "difference_ci_95": (diff_lower, diff_upper),
        "is_significant": is_significant,
        "p_value_proxy": "< 0.05" if is_significant else ">= 0.05",
    }

# Monkey-patch for clean API usage in ReportGenerator
StatisticalTester.compare_to_results = _patch_compare_to_results

