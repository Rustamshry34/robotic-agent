"""
FastAPI application entry point for the self-verifying robotic AI agent.

Endpoints:
    GET  /health                    - Service health check
    GET  /health/db                 - Database connectivity check
    POST /tasks/run                 - Execute a robot task
    GET  /tasks/{task_id}           - Get task execution details
    POST /evaluation/run            - Run baseline comparison experiment
    GET  /evaluation/{eval_id}      - Get evaluation results
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.database import (
    check_database_connection,
    shutdown_db,
    startup_db,
)
from app.agent.executor import (
    RobotExecutor,
    SimulatedMCPToolInvoker,
)
from app.agent.planner import create_planner


logger = logging.getLogger(__name__)

settings = get_settings()


# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------

logging.basicConfig(
    level=settings.logging_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Startup:
        - Initialize database connection
        - Create tables if enabled
        - Warm up planner and executor

    Shutdown:
        - Dispose database engine
    """
    logger.info("Starting up %s", settings.app_name)
    logger.info("Environment: %s", settings.environment.value)
    logger.info("Version: %s", settings.agent_version)

    await startup_db()

    # Warm up shared instances
    app.state.tool_invoker = SimulatedMCPToolInvoker()
    app.state.executor = RobotExecutor(settings=settings)

    logger.info("Application startup complete.")

    yield

    logger.info("Shutting down %s", settings.app_name)
    await shutdown_db()
    logger.info("Application shutdown complete.")


# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version=settings.agent_version,
    description=(
        "MCP-backed, self-verifying Robotic AI Agent. "
        "Plans robot tasks, executes via MCP tools, verifies results, "
        "and recovers from errors using experience memory."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------


class TaskRunRequest(BaseModel):
    instruction: str = Field(
        min_length=1,
        max_length=2000,
        description="Natural-language task instruction.",
        examples=[
            "Take the red cup on the table, put it on the kitchen counter, "
            "and verify the process is complete."
        ],
    )

    task_name: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Optional task name. Auto-generated if not provided.",
    )

    baseline: Optional[str] = Field(
        default=None,
        description=(
            "Baseline override: baseline_a, baseline_b, baseline_c, "
            "or full_system."
        ),
    )

    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional metadata to attach to the task run.",
    )


class TaskRunResponse(BaseModel):
    task_id: Optional[str] = None
    task_name: str
    success: bool
    status: str
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    recovery_attempts: int = 0
    retry_count: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    duration_ms: int = 0


class EvaluationRequest(BaseModel):
    instruction: str = Field(
        min_length=1,
        max_length=2000,
        description="Task instruction to evaluate across baselines.",
    )

    baselines: list[str] = Field(
        default=[
            "baseline_a",
            "baseline_b",
            "baseline_c",
            "full_system",
        ],
        description="List of baselines to compare.",
    )

    iterations: int = Field(
        default=1,
        ge=1,
        le=50,
        description="Number of iterations per baseline.",
    )

    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the evaluation run.",
    )


class BaselineResult(BaseModel):
    baseline: str
    iteration: int
    success: bool
    status: str
    error_code: Optional[str] = None
    recovery_attempts: int = 0
    retry_count: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    duration_ms: int = 0


class EvaluationResponse(BaseModel):
    evaluation_id: str
    instruction: str
    total_runs: int
    results: list[BaselineResult] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str


class DBHealthResponse(BaseModel):
    status: str
    database_connected: bool


# ------------------------------------------------------------------
# Health endpoints
# ------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.agent_version,
        environment=settings.environment.value,
    )


@app.get(
    "/health/db",
    response_model=DBHealthResponse,
    tags=["health"],
)
async def database_health_check() -> DBHealthResponse:
    connected = await check_database_connection()

    return DBHealthResponse(
        status="ok" if connected else "degraded",
        database_connected=connected,
    )


# ------------------------------------------------------------------
# Task endpoints
# ------------------------------------------------------------------


@app.post(
    "/tasks/run",
    response_model=TaskRunResponse,
    status_code=status.HTTP_200_OK,
    tags=["tasks"],
)
async def run_task(request: TaskRunRequest) -> TaskRunResponse:
    """
    Execute a robot task from a natural-language instruction.

    The system will:
    1. Plan the task using the planner
    2. Execute steps via MCP tools
    3. Verify critical actions
    4. Attempt recovery on failure (full_system baseline)
    """
    executor: RobotExecutor = app.state.executor
    tool_invoker: SimulatedMCPToolInvoker = app.state.tool_invoker

    try:
        result = await executor.execute_task(
            instruction=request.instruction,
            tool_invoker=tool_invoker,
            task_name=request.task_name,
            baseline=request.baseline,
            metadata=request.metadata,
        )
    except Exception as exc:
        logger.exception("Task execution failed unexpectedly.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task execution failed: {exc}",
        ) from exc

    return TaskRunResponse(
        task_id=result.task_id,
        task_name=result.task_name,
        success=result.success,
        status=result.status,
        error_code=result.error_code,
        error_message=result.error_message,
        recovery_attempts=result.recovery_attempts,
        retry_count=result.retry_count,
        metrics=result.metrics,
        duration_ms=result.duration_ms,
    )


# ------------------------------------------------------------------
# Evaluation endpoints
# ------------------------------------------------------------------


@app.post(
    "/evaluation/run",
    response_model=EvaluationResponse,
    status_code=status.HTTP_200_OK,
    tags=["evaluation"],
)
async def run_evaluation(request: EvaluationRequest) -> EvaluationResponse:
    """
    Run a baseline comparison experiment.

    Executes the same task instruction across multiple baselines
    and returns per-baseline results with summary statistics.
    """
    evaluation_id = str(uuid.uuid4())
    executor: RobotExecutor = app.state.executor
    tool_invoker: SimulatedMCPToolInvoker = app.state.tool_invoker

    results: list[BaselineResult] = []

    for baseline in request.baselines:
        for iteration in range(1, request.iterations + 1):
            logger.info(
                "Evaluation run: baseline=%s iteration=%d/%d",
                baseline,
                iteration,
                request.iterations,
            )

            try:
                result = await executor.execute_task(
                    instruction=request.instruction,
                    tool_invoker=tool_invoker,
                    baseline=baseline,
                    metadata={
                        **(request.metadata or {}),
                        "evaluation_id": evaluation_id,
                        "baseline": baseline,
                        "iteration": iteration,
                    },
                )

                results.append(
                    BaselineResult(
                        baseline=baseline,
                        iteration=iteration,
                        success=result.success,
                        status=result.status,
                        error_code=result.error_code,
                        recovery_attempts=result.recovery_attempts,
                        retry_count=result.retry_count,
                        metrics=result.metrics,
                        duration_ms=result.duration_ms,
                    )
                )

            except Exception as exc:
                logger.exception(
                    "Evaluation run failed: baseline=%s iteration=%d",
                    baseline,
                    iteration,
                )

                results.append(
                    BaselineResult(
                        baseline=baseline,
                        iteration=iteration,
                        success=False,
                        status="error",
                        error_code="EVALUATION_EXCEPTION",
                    )
                )

    summary = _build_evaluation_summary(results)

    return EvaluationResponse(
        evaluation_id=evaluation_id,
        instruction=request.instruction,
        total_runs=len(results),
        results=results,
        summary=summary,
    )


def _build_evaluation_summary(
    results: list[BaselineResult],
) -> dict[str, Any]:
    """
    Build summary statistics for evaluation results.
    """
    if not results:
        return {}

    baselines = sorted({r.baseline for r in results})
    summary: dict[str, Any] = {}

    for baseline in baselines:
        baseline_results = [r for r in results if r.baseline == baseline]
        total = len(baseline_results)
        successes = sum(1 for r in baseline_results if r.success)
        failures = total - successes

        avg_duration = (
            sum(r.duration_ms for r in baseline_results) / total
            if total > 0
            else 0
        )

        total_recoveries = sum(
            r.recovery_attempts for r in baseline_results
        )
        total_retries = sum(r.retry_count for r in baseline_results)

        summary[baseline] = {
            "total_runs": total,
            "success_count": successes,
            "failure_count": failures,
            "success_rate": successes / total if total > 0 else 0.0,
            "average_duration_ms": round(avg_duration, 2),
            "total_recovery_attempts": total_recoveries,
            "total_retries": total_retries,
        }

    return summary


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug,
        log_level=settings.logging_level.lower(),
    )
