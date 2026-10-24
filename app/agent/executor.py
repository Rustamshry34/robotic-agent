"""
Executor for the self-verifying robotic AI agent.

This module executes a structured RobotPlan using MCP tools.

Responsibilities:
- execute plan steps
- call MCP tools
- verify critical actions
- handle retries and recovery
- log task/step results to PostgreSQL
- store experiences in ChromaDB
- produce execution metrics
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.agent.planner import (
    DEFAULT_TOOL_SPECS,
    FailureStrategy,
    PlanStep,
    PlanStepType,
    RobotPlan,
    RobotPlanner,
    create_planner,
)

from memory.chroma_client import ChromaMemoryClient, get_chroma_memory
from memory.models import (
    BaselineName,
    ExperienceOutcome,
    MemoryExperience,
    StepStatus,
    StepType as DBStepType,
    TaskMetric,
    TaskRun,
    TaskStatus,
    TaskStep as DBTaskStep,
)

from mcp_server.tools import SimulatedRobotBackend


logger = logging.getLogger(__name__)


__all__ = [
    "ToolInvoker",
    "SimulatedMCPToolInvoker",
    "ExecutorConfig",
    "StepExecutionResult",
    "ExecutionResult",
    "RobotExecutor",
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def duration_ms(start: datetime, end: datetime) -> int:
    return int((end - start).total_seconds() * 1000)


def _normalize_tool_result(result: Any) -> dict[str, Any]:
    """
    Normalize tool outputs into a standard ToolResult dictionary.

    Expected shape:
        {
            "success": bool,
            "message": str,
            "data": dict,
            "error_code": str | None
        }
    """
    if hasattr(result, "model_dump"):
        result = result.model_dump()

    if isinstance(result, dict):
        return {
            "success": bool(result.get("success", False)),
            "message": str(result.get("message", "")),
            "data": result.get("data") or {},
            "error_code": result.get("error_code"),
        }

    return {
        "success": False,
        "message": str(result),
        "data": {},
        "error_code": "UNKNOWN_TOOL_RESULT",
    }


def _to_baseline_enum(value: Optional[str]) -> BaselineName:
    if not value:
        return BaselineName.FULL_SYSTEM

    try:
        return BaselineName(value)
    except ValueError:
        return BaselineName.FULL_SYSTEM


def _to_db_step_type(value: PlanStepType) -> DBStepType:
    return DBStepType(value.value)


def _state_summary_text(state: dict[str, Any]) -> str:
    text = json.dumps(state, ensure_ascii=False, default=str)
    return text[:2000]


# ----------------------------------------------------------------------
# Tool invoker abstraction
# ----------------------------------------------------------------------


@runtime_checkable
class ToolInvoker(Protocol):
    """
    Abstraction for calling MCP tools.

    Implementations:
    - SimulatedMCPToolInvoker
    - future MCP stdio client
    - future ROS 2 bridge client
    """

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        ...

    def list_tools(self) -> list[dict[str, Any]]:
        ...


class SimulatedMCPToolInvoker:
    """
    Local tool invoker backed by SimulatedRobotBackend.

    This is useful for:
    - local development
    - CI tests
    - evaluation smoke tests
    """

    def __init__(self, backend: Optional[SimulatedRobotBackend] = None) -> None:
        self._backend = backend or SimulatedRobotBackend()

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        method_name = tool_name.removeprefix("robot_")
        method = getattr(self._backend, method_name, None)

        if method is None or not callable(method):
            return {
                "success": False,
                "message": f"Tool '{tool_name}' not found.",
                "data": {},
                "error_code": "UNKNOWN_TOOL",
            }

        try:
            return await asyncio.to_thread(method, **arguments)
        except TypeError as exc:
            return {
                "success": False,
                "message": f"Invalid arguments for tool '{tool_name}': {exc}",
                "data": {},
                "error_code": "INVALID_TOOL_ARGUMENTS",
            }
        except Exception as exc:
            logger.exception("Simulated tool execution failed.")
            return {
                "success": False,
                "message": str(exc),
                "data": {},
                "error_code": "TOOL_EXECUTION_EXCEPTION",
            }

    def list_tools(self) -> list[dict[str, Any]]:
        return DEFAULT_TOOL_SPECS


# ----------------------------------------------------------------------
# Executor config and result schemas
# ----------------------------------------------------------------------


class ExecutorConfig(BaseModel):
    refresh_state_after_action: bool = True
    store_step_experiences: bool = True

    max_recovery_attempts: int = Field(default=3, ge=0)
    step_timeout_seconds: float = Field(default=30.0, gt=0)

    recovery_baselines: tuple[str, ...] = (
        "full_system",
    )

    @staticmethod
    def from_settings(settings: Settings) -> "ExecutorConfig":
        return ExecutorConfig(
            refresh_state_after_action=True,
            store_step_experiences=settings.agent_enable_memory,
            max_recovery_attempts=settings.agent_max_action_retries,
            step_timeout_seconds=float(settings.mcp_timeout_seconds),
            recovery_baselines=("full_system",),
        )


class StepExecutionResult(BaseModel):
    step_id: int
    tool_name: str

    success: bool
    attempts: int = 1

    verified: Optional[bool] = None

    output: dict[str, Any] = Field(default_factory=dict)

    error_code: Optional[str] = None
    error_message: Optional[str] = None

    started_at: datetime
    finished_at: datetime
    duration_ms: int


class ExecutionResult(BaseModel):
    task_id: Optional[str] = None
    task_name: str

    success: bool
    status: str

    error_code: Optional[str] = None
    error_message: Optional[str] = None

    current_state: dict[str, Any] = Field(default_factory=dict)

    steps: list[StepExecutionResult] = Field(default_factory=list)

    recovery_attempts: int = 0
    retry_count: int = 0

    metrics: dict[str, float] = Field(default_factory=dict)

    started_at: datetime
    finished_at: datetime
    duration_ms: int


# ----------------------------------------------------------------------
# Executor
# ----------------------------------------------------------------------


class RobotExecutor:
    """
    Executes robot plans produced by RobotPlanner.
    """

    def __init__(
        self,
        planner: Optional[RobotPlanner] = None,
        settings: Optional[Settings] = None,
        memory_client: Optional[ChromaMemoryClient] = None,
        config: Optional[ExecutorConfig] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.config = config or ExecutorConfig.from_settings(self.settings)

        self.planner = planner or create_planner(settings=self.settings)

        if memory_client is None and self.settings.agent_enable_memory:
            try:
                self.memory_client = get_chroma_memory()
            except Exception:
                logger.exception(
                    "Could not initialize ChromaMemoryClient. "
                    "Experience storage will be disabled."
                )
                self.memory_client = None
        else:
            self.memory_client = memory_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute_task(
        self,
        instruction: str,
        tool_invoker: ToolInvoker,
        task_name: Optional[str] = None,
        baseline: Optional[str] = None,
        current_state: Optional[dict[str, Any]] = None,
        db_session: Optional[AsyncSession] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Plan and execute a task from a natural-language instruction.
        """
        task_planner = self.planner

        if baseline is not None:
            task_planner = create_planner(
                settings=self.settings,
                memory_client=self.memory_client,
                baseline=baseline,
            )

        available_tools = DEFAULT_TOOL_SPECS

        if hasattr(tool_invoker, "list_tools"):
            try:
                available_tools = tool_invoker.list_tools()
            except Exception:
                logger.exception(
                    "tool_invoker.list_tools() failed. "
                    "Using default tool specs."
                )

        plan = await task_planner.plan(
            task_instruction=instruction,
            task_name=task_name,
            current_state=current_state,
            available_tools=available_tools,
        )

        return await self.execute_plan(
            plan=plan,
            task_instruction=instruction,
            tool_invoker=tool_invoker,
            current_state=current_state,
            db_session=db_session,
            metadata=metadata,
            planner=task_planner,
        )

    async def execute_plan(
        self,
        plan: RobotPlan,
        task_instruction: str,
        tool_invoker: ToolInvoker,
        current_state: Optional[dict[str, Any]] = None,
        db_session: Optional[AsyncSession] = None,
        metadata: Optional[dict[str, Any]] = None,
        planner: Optional[RobotPlanner] = None,
    ) -> ExecutionResult:
        """
        Execute an already-generated RobotPlan.
        """
        started_at = utcnow()

        active_plan = plan
        pending_steps = list(active_plan.steps)

        step_results: list[StepExecutionResult] = []

        completed_steps = 0
        failed_steps = 0
        retry_count = 0
        recovery_attempts = 0

        error_code: Optional[str] = None
        error_message: Optional[str] = None

        completed_successfully = False

        task_run = await self._create_task_run(
            db_session=db_session,
            plan=active_plan,
            instruction=task_instruction,
            metadata=metadata,
        )

        if current_state is None:
            current_state = await self._safe_get_state(tool_invoker)

        while pending_steps:
            step = pending_steps.pop(0)
            step_index = len(step_results) + 1

            step_result = await self._execute_step(
                step=step,
                step_index=step_index,
                plan=active_plan,
                tool_invoker=tool_invoker,
                current_state=current_state,
                db_session=db_session,
                task_run=task_run,
            )

            step_results.append(step_result)
            retry_count += max(0, step_result.attempts - 1)

            if self.config.refresh_state_after_action:
                current_state = await self._safe_get_state(
                    tool_invoker,
                    fallback=current_state,
                )

            if step_result.success:
                completed_steps += 1

                await self._store_step_experience(
                    step=step,
                    step_result=step_result,
                    outcome=ExperienceOutcome.SUCCESS,
                    current_state=current_state,
                    task_id=task_run.id if task_run else None,
                    task_name=active_plan.task_name,
                    step_index=step_index,
                    db_session=db_session,
                )

                continue

            failed_steps += 1

            can_recover = (
                planner is not None
                and active_plan.baseline in self.config.recovery_baselines
                and recovery_attempts < self.config.max_recovery_attempts
            )

            resolution = (
                "Generated recovery plan."
                if can_recover
                else None
            )

            await self._store_step_experience(
                step=step,
                step_result=step_result,
                outcome=ExperienceOutcome.FAILURE,
                current_state=current_state,
                task_id=task_run.id if task_run else None,
                task_name=active_plan.task_name,
                step_index=step_index,
                db_session=db_session,
                resolution=resolution,
            )

            if can_recover:
                try:
                    recovery_plan = await planner.create_recovery_plan(
                        task_instruction=task_instruction,
                        failed_plan=active_plan,
                        failed_step=step,
                        error_code=step_result.error_code or "UNKNOWN_ERROR",
                        error_message=step_result.error_message,
                        current_state=current_state,
                        task_name=active_plan.task_name,
                    )

                    recovery_attempts += 1
                    active_plan = recovery_plan
                    pending_steps = list(recovery_plan.steps)

                    if db_session and task_run:
                        task_run.status = TaskStatus.RECOVERING
                        task_run.metadata_ = {
                            **task_run.metadata_,
                            "recovery_attempts": recovery_attempts,
                        }
                        await db_session.flush()

                    logger.info(
                        "Recovery plan generated. recovery_attempts=%s",
                        recovery_attempts,
                    )

                    continue

                except Exception:
                    logger.exception("Recovery planning failed.")

                    error_code = step_result.error_code or "RECOVERY_FAILED"
                    error_message = step_result.error_message
                    break

            error_code = step_result.error_code
            error_message = step_result.error_message
            break

        else:
            completed_successfully = True

        finished_at = utcnow()
        execution_duration_ms = duration_ms(started_at, finished_at)

        status = (
            TaskStatus.COMPLETED.value
            if completed_successfully
            else TaskStatus.FAILED.value
        )

        metrics = {
            "task_duration_ms": float(execution_duration_ms),
            "executed_steps": float(len(step_results)),
            "completed_steps": float(completed_steps),
            "failed_steps": float(failed_steps),
            "retry_count": float(retry_count),
            "recovery_attempts": float(recovery_attempts),
        }

        await self._finalize_task_run(
            db_session=db_session,
            task_run=task_run,
            success=completed_successfully,
            status=status,
            error_code=error_code,
            error_message=error_message,
            completed_steps=completed_steps,
            failed_steps=failed_steps,
            retry_count=retry_count,
            recovery_attempts=recovery_attempts,
            executed_steps=len(step_results),
            finished_at=finished_at,
            duration_ms=execution_duration_ms,
        )

        return ExecutionResult(
            task_id=str(task_run.id) if task_run else None,
            task_name=active_plan.task_name,
            success=completed_successfully,
            status=status,
            error_code=error_code,
            error_message=error_message,
            current_state=current_state,
            steps=step_results,
            recovery_attempts=recovery_attempts,
            retry_count=retry_count,
            metrics=metrics,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=execution_duration_ms,
        )

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    async def _execute_step(
        self,
        step: PlanStep,
        step_index: int,
        plan: RobotPlan,
        tool_invoker: ToolInvoker,
        current_state: dict[str, Any],
        db_session: Optional[AsyncSession],
        task_run: Optional[TaskRun],
    ) -> StepExecutionResult:
        started_at = utcnow()

        db_step = await self._create_step_record(
            db_session=db_session,
            task_run=task_run,
            step=step,
            step_index=step_index,
        )

        max_attempts = 1

        if step.on_failure == FailureStrategy.RETRY:
            max_attempts = step.max_retries + 1

        attempts = 0
        verified: Optional[bool] = None

        last_result: dict[str, Any] = {
            "success": False,
            "message": "Step did not execute.",
            "data": {},
            "error_code": "STEP_NOT_EXECUTED",
        }

        while attempts < max_attempts:
            attempts += 1

            if db_session and db_step:
                db_step.status = (
                    StepStatus.RUNNING
                    if attempts == 1
                    else StepStatus.RETRYING
                )
                db_step.retry_count = attempts - 1
                await db_session.flush()

            raw_result = await self._call_tool(
                tool_invoker=tool_invoker,
                tool_name=step.tool_name,
                arguments=step.input_payload,
            )

            result = _normalize_tool_result(raw_result)

            if result["success"]:
                if step.step_type == PlanStepType.VERIFICATION:
                    verified = bool(
                        (result.get("data") or {}).get("verified", False)
                    )

                    if not verified:
                        result = {
                            "success": False,
                            "message": result.get(
                                "message",
                                "Verification failed.",
                            ),
                            "data": result.get("data", {}),
                            "error_code": "VERIFICATION_FAILED",
                        }

                elif plan.requires_verification and step.verification_tool:
                    verification_raw = await self._call_tool(
                        tool_invoker=tool_invoker,
                        tool_name=step.verification_tool,
                        arguments=step.verification_payload,
                    )

                    verification_result = _normalize_tool_result(
                        verification_raw
                    )

                    verified = bool(
                        (verification_result.get("data") or {}).get(
                            "verified",
                            False,
                        )
                    )

                    if not verification_result["success"]:
                        result = {
                            **verification_result,
                            "error_code": (
                                verification_result.get("error_code")
                                or "VERIFICATION_TOOL_FAILED"
                            ),
                        }
                    elif not verified:
                        result = {
                            "success": False,
                            "message": verification_result.get(
                                "message",
                                "Verification failed.",
                            ),
                            "data": verification_result.get("data", {}),
                            "error_code": "VERIFICATION_FAILED",
                        }

            if result["success"]:
                finished_at = utcnow()
                step_duration_ms = duration_ms(started_at, finished_at)

                if db_session and db_step:
                    db_step.status = StepStatus.SUCCESS
                    db_step.output_payload = result
                    db_step.verification_payload = step.verification_payload
                    db_step.verified = verified
                    db_step.finished_at = finished_at
                    db_step.duration_ms = step_duration_ms
                    await db_session.flush()

                return StepExecutionResult(
                    step_id=step.step_id,
                    tool_name=step.tool_name,
                    success=True,
                    attempts=attempts,
                    verified=verified,
                    output=result,
                    error_code=None,
                    error_message=None,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_ms=step_duration_ms,
                )

            last_result = result

            if db_session and db_step:
                db_step.output_payload = result
                db_step.error_code = result.get("error_code")
                db_step.error_message = result.get("message")
                db_step.status = (
                    StepStatus.RETRYING
                    if attempts < max_attempts
                    else StepStatus.FAILED
                )
                await db_session.flush()

        finished_at = utcnow()
        step_duration_ms = duration_ms(started_at, finished_at)

        if db_session and db_step:
            db_step.status = StepStatus.FAILED
            db_step.output_payload = last_result
            db_step.error_code = last_result.get("error_code")
            db_step.error_message = last_result.get("message")
            db_step.finished_at = finished_at
            db_step.duration_ms = step_duration_ms
            await db_session.flush()

        return StepExecutionResult(
            step_id=step.step_id,
            tool_name=step.tool_name,
            success=False,
            attempts=attempts,
            verified=verified,
            output=last_result,
            error_code=last_result.get("error_code"),
            error_message=last_result.get("message"),
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=step_duration_ms,
        )

    async def _call_tool(
        self,
        tool_invoker: ToolInvoker,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(
                tool_invoker.call_tool(tool_name, arguments or {}),
                timeout=self.config.step_timeout_seconds,
            )
        except asyncio.TimeoutError:
            return {
                "success": False,
                "message": (
                    f"Tool '{tool_name}' timed out after "
                    f"{self.config.step_timeout_seconds} seconds."
                ),
                "data": {},
                "error_code": "TOOL_TIMEOUT",
            }
        except Exception as exc:
            logger.exception("Tool invocation failed.")
            return {
                "success": False,
                "message": str(exc),
                "data": {},
                "error_code": "TOOL_INVOCATION_EXCEPTION",
            }

    async def _safe_get_state(
        self,
        tool_invoker: ToolInvoker,
        fallback: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        fallback = fallback or {}

        try:
            raw_state = await self._call_tool(
                tool_invoker=tool_invoker,
                tool_name="robot_get_state",
                arguments={},
            )

            state_result = _normalize_tool_result(raw_state)

            if state_result["success"]:
                return state_result.get("data") or fallback

        except Exception:
            logger.exception("Could not refresh robot state.")

        return fallback

    # ------------------------------------------------------------------
    # PostgreSQL logging
    # ------------------------------------------------------------------

    async def _create_task_run(
        self,
        db_session: Optional[AsyncSession],
        plan: RobotPlan,
        instruction: str,
        metadata: Optional[dict[str, Any]],
    ) -> Optional[TaskRun]:
        if db_session is None:
            return None

        try:
            baseline_enum = _to_baseline_enum(plan.baseline)

            task_metadata = {
                **(metadata or {}),
                "plan_rationale": plan.rationale,
                "requires_verification": plan.requires_verification,
                "requires_memory": plan.requires_memory,
                "final_verification": plan.final_verification,
            }

            task_run = TaskRun(
                task_name=plan.task_name,
                instruction=instruction,
                status=TaskStatus.EXECUTING,
                baseline=baseline_enum,
                robot_backend="simulated",
                environment=self.settings.environment.value,
                agent_version=self.settings.agent_version,
                git_commit=self.settings.git_commit,
                total_steps=len(plan.steps),
                started_at=utcnow(),
                metadata_=task_metadata,
            )

            db_session.add(task_run)
            await db_session.flush()

            return task_run

        except Exception:
            logger.exception("Could not create TaskRun record.")
            return None

    async def _create_step_record(
        self,
        db_session: Optional[AsyncSession],
        task_run: Optional[TaskRun],
        step: PlanStep,
        step_index: int,
    ) -> Optional[DBTaskStep]:
        if db_session is None or task_run is None:
            return None

        try:
            step_metadata = {
                "description": step.description,
                "on_failure": step.on_failure.value,
                "max_retries": step.max_retries,
                "verification_tool": step.verification_tool,
            }

            db_step = DBTaskStep(
                task_id=task_run.id,
                step_index=step_index,
                action=step.tool_name,
                step_type=_to_db_step_type(step.step_type),
                status=StepStatus.PENDING,
                input_payload=step.input_payload,
                expected_state=step.expected_state,
                metadata_=step_metadata,
            )

            db_session.add(db_step)
            await db_session.flush()

            return db_step

        except Exception:
            logger.exception("Could not create TaskStep record.")
            return None

    async def _finalize_task_run(
        self,
        db_session: Optional[AsyncSession],
        task_run: Optional[TaskRun],
        success: bool,
        status: str,
        error_code: Optional[str],
        error_message: Optional[str],
        completed_steps: int,
        failed_steps: int,
        retry_count: int,
        recovery_attempts: int,
        executed_steps: int,
        finished_at: datetime,
        duration_ms_value: int,
    ) -> None:
        if db_session is None or task_run is None:
            return

        try:
            task_run.status = TaskStatus(status)
            task_run.success = success

            task_run.completed_steps = completed_steps
            task_run.failed_steps = failed_steps
            task_run.retry_count = retry_count
            task_run.total_steps = executed_steps

            task_run.error_code = error_code
            task_run.error_message = error_message

            task_run.finished_at = finished_at
            task_run.duration_ms = duration_ms_value

            task_run.metadata_ = {
                **task_run.metadata_,
                "recovery_attempts": recovery_attempts,
                "executed_steps": executed_steps,
            }

            await db_session.flush()

            metrics = [
                ("task_duration_ms", float(duration_ms_value)),
                ("executed_steps", float(executed_steps)),
                ("completed_steps", float(completed_steps)),
                ("failed_steps", float(failed_steps)),
                ("retry_count", float(retry_count)),
                ("recovery_attempts", float(recovery_attempts)),
            ]

            for metric_name, metric_value in metrics:
                db_session.add(
                    TaskMetric(
                        task_id=task_run.id,
                        metric_name=metric_name,
                        metric_value=metric_value,
                    )
                )

            await db_session.flush()

        except Exception:
            logger.exception("Could not finalize TaskRun record.")

    # ------------------------------------------------------------------
    # ChromaDB experience storage
    # ------------------------------------------------------------------

    async def _store_step_experience(
        self,
        step: PlanStep,
        step_result: StepExecutionResult,
        outcome: ExperienceOutcome,
        current_state: dict[str, Any],
        task_id: Optional[Any],
        task_name: str,
        step_index: int,
        db_session: Optional[AsyncSession],
        resolution: Optional[str] = None,
    ) -> None:
        if not self.memory_client or not self.config.store_step_experiences:
            return

        if step.step_type == PlanStepType.OBSERVATION:
            return

        try:
            experience_id = await asyncio.to_thread(
                self.memory_client.add_experience,
                action=step.tool_name,
                outcome=outcome.value,
                state_summary=current_state,
                error_code=step_result.error_code,
                error_message=step_result.error_message,
                resolution=resolution,
                task_id=str(task_id) if task_id else None,
                task_name=task_name,
                step_index=step_index,
            )

            if db_session is not None and task_id is not None:
                experience = MemoryExperience(
                    id=experience_id,
                    task_id=task_id,
                    action=step.tool_name,
                    outcome=outcome,
                    error_code=step_result.error_code or "NONE",
                    task_name=task_name,
                    step_index=step_index,
                    state_summary=_state_summary_text(current_state),
                    error_message=step_result.error_message,
                    resolution=resolution,
                    chroma_synced=True,
                    chroma_synced_at=utcnow(),
                    metadata_={
                        "tool_name": step.tool_name,
                        "step_id": step.step_id,
                        "attempts": step_result.attempts,
                    },
                )

                db_session.add(experience)
                await db_session.flush()

        except Exception:
            logger.exception("Could not store step experience.")
