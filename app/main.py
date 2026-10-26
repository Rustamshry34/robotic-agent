"""
FastAPI application entry point for the self-verifying robotic AI agent.

Endpoints:
    GET  /health                    - Service health check
    GET  /health/db                 - Database connectivity check
    POST /tasks/run                 - Execute a robot task
    POST /evaluation/run            - Run baseline comparison experiment (via evaluation.baselines)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agent.executor import RobotExecutor, SimulatedMCPToolInvoker
from app.core.config import get_settings
from app.core.database import (
    check_database_connection,
    shutdown_db,
    startup_db,
)
from evaluation.baselines import (
    AVAILABLE_BASELINES,
    BaselineExperiment,
    ExperimentResult,
)


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

    task_name: str | None = Field(
        default=None,
        max_length=255,
        description="Optional task name. Auto-generated if not provided.",
    )

    baseline: str | None = Field(
        default=None,
        description=(
            "Baseline override: baseline_a, baseline_b, baseline_c, "
            "or full_system."
        ),
    )

    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata to attach to the task run.",
    )


class TaskRunResponse(BaseModel):
    task_id: str | None = None
    task_name: str
    success: bool
    status: str
    error_code: str | None = None
    error_message: str | None = None
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
        default=list(AVAILABLE_BASELINES),
        description="List of baselines to compare.",
    )

    iterations: int = Field(
        default=1,
        ge=1,
        le=50,
        description="Number of iterations per baseline.",
    )

    seed: int | None = Field(
        default=None,
        description="Optional random seed for reproducibility.",
    )

    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata for the evaluation run.",
    )


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
    except Exception:
        logger.exception("Task execution failed unexpectedly.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Task execution failed. Check logs for details.",
        ) from None

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
    response_model=ExperimentResult,
    status_code=status.HTTP_200_OK,
    tags=["evaluation"],
)
async def run_evaluation(request: EvaluationRequest) -> ExperimentResult:
    """
    Run a baseline comparison experiment.

    Delegates to evaluation.baselines.BaselineExperiment for:
    - Reproducible seeding
    - Rich metric collection (verified steps, error distribution, etc.)
    - Centralized summary generation
    """
    
    executor: RobotExecutor = app.state.executor  # ← app.state'ten al
    
    try:
        experiment = BaselineExperiment(
            instruction=request.instruction,
            baselines=request.baselines,
            iterations=request.iterations,
            seed=request.seed,
            metadata=request.metadata,
            executor=executor,  # ← YENİ: executor'ı inject et
        )

        result = await experiment.run()
        return result

    except ValueError as exc:
        # Catch invalid baseline names or other validation errors from BaselineExperiment
        logger.warning("Evaluation request validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Evaluation experiment failed unexpectedly.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evaluation experiment failed. Check logs for details.",
        ) from exc


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
