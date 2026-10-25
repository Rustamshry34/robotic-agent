"""
PostgreSQL SQLAlchemy models for the self-verifying robotic AI agent.

These models store:
- task execution runs
- step-level agent actions
- verification results
- errors and recovery attempts
- evaluation metrics
- relational audit records for experiences stored in ChromaDB
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SQLAlchemyEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy.types import JSON


JSONType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models.
    """

    pass


def _enum_column(enum_cls: type[enum.Enum]) -> SQLAlchemyEnum:
    """
    Create a portable SQLAlchemy Enum column.

    native_enum=False avoids painful PostgreSQL enum migrations.
    """
    return SQLAlchemyEnum(
        enum_cls,
        native_enum=False,
        length=32,
        values_callable=lambda obj: [member.value for member in obj],
    )


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"
    RECOVERED = "recovered"


class StepType(str, enum.Enum):
    PLAN = "plan"
    ACTION = "action"
    OBSERVATION = "observation"
    VERIFICATION = "verification"
    RECOVERY = "recovery"


class BaselineName(str, enum.Enum):
    BASELINE_A = "baseline_a"
    BASELINE_B = "baseline_b"
    BASELINE_C = "baseline_c"
    FULL_SYSTEM = "full_system"


class ExperienceOutcome(str, enum.Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RECOVERED = "recovered"


class TaskRun(Base):
    """
    A single end-to-end task execution.

    Example task:
        "Take the red cup on the table, put it on the kitchen counter,
         and verify the process is complete."
    """

    __tablename__ = "task_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )

    task_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    instruction: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        _enum_column(TaskStatus),
        default=TaskStatus.PENDING.value,
        nullable=False,
        index=True,
    )

    baseline: Mapped[str] = mapped_column(
        _enum_column(BaselineName),
        default=BaselineName.FULL_SYSTEM.value,
        nullable=False,
        index=True,
    )

    robot_backend: Mapped[str] = mapped_column(
        String(64),
        default="simulated",
        nullable=False,
    )

    environment: Mapped[str] = mapped_column(
        String(64),
        default="local",
        nullable=False,
    )

    agent_version: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    git_commit: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    success: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )

    total_steps: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    completed_steps: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    failed_steps: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    retry_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    prompt_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    completion_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    total_tokens: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    estimated_cost_usd: Mapped[float | None] = mapped_column(
        Numeric(12, 6),
        nullable=True,
    )

    error_code: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONType,
        default=dict,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    steps: Mapped[list[TaskStep]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="TaskStep.step_index",
    )

    metrics: Mapped[list[TaskMetric]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    experiences: Mapped[list[MemoryExperience]] = relationship(
        back_populates="task",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_task_runs_status_created_at", "status", "created_at"),
        Index("ix_task_runs_baseline_status", "baseline", "status"),
        Index("ix_task_runs_task_name_created_at", "task_name", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<TaskRun id={self.id} task_name={self.task_name} "
            f"status={self.status} baseline={self.baseline}>"
        )


class TaskStep(Base):
    """
    A single step inside a task execution.

    Steps may include:
    - planning
    - robot action
    - observation
    - verification
    - recovery
    """

    __tablename__ = "task_steps"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )

    task_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("task_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    step_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )

    action: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    step_type: Mapped[str] = mapped_column(
        _enum_column(StepType),
        default=StepType.ACTION.value,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        _enum_column(StepStatus),
        default=StepStatus.PENDING.value,
        nullable=False,
        index=True,
    )

    input_payload: Mapped[dict[str, Any]] = mapped_column(
        JSONType,
        default=dict,
        nullable=False,
    )

    output_payload: Mapped[dict[str, Any]] = mapped_column(
        JSONType,
        default=dict,
        nullable=False,
    )

    verification_payload: Mapped[dict[str, Any]] = mapped_column(
        JSONType,
        default=dict,
        nullable=False,
    )

    expected_state: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    actual_state: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    verified: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
    )

    error_code: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    resolution: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    retry_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONType,
        default=dict,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    task: Mapped[TaskRun] = relationship(
        back_populates="steps",
    )

    __table_args__ = (
        Index("ix_task_steps_task_id_step_index", "task_id", "step_index"),
        Index("ix_task_steps_action_status", "action", "status"),
        Index("ix_task_steps_error_code", "error_code"),
    )

    def __repr__(self) -> str:
        return (
            f"<TaskStep id={self.id} task_id={self.task_id} "
            f"action={self.action} status={self.status}>"
        )


class TaskMetric(Base):
    """
    Evaluation metrics for a task run.

    Examples:
        task_success_rate
        post_error_recovery_rate
        unsafe_action_rate
        llm_token_cost
        steps_per_task
        duration_ms
    """

    __tablename__ = "task_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )

    task_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("task_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    metric_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    metric_value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    unit: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
    )

    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONType,
        default=dict,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    task: Mapped[TaskRun] = relationship(
        back_populates="metrics",
    )

    __table_args__ = (
        Index("ix_task_metrics_task_id_metric_name", "task_id", "metric_name"),
    )

    def __repr__(self) -> str:
        return (
            f"<TaskMetric id={self.id} task_id={self.task_id} "
            f"metric_name={self.metric_name} metric_value={self.metric_value}>"
        )


class MemoryExperience(Base):
    """
    Relational audit table for experiences stored in ChromaDB.

    ChromaDB is used for semantic retrieval.
    PostgreSQL is used for traceability, joins, evaluation, and auditing.
    """

    __tablename__ = "memory_experiences"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )

    task_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey("task_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    step_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey("task_steps.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    action: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    outcome: Mapped[str] = mapped_column(
        _enum_column(ExperienceOutcome),
        nullable=False,
        index=True,
    )

    error_code: Mapped[str] = mapped_column(
        String(64),
        default="NONE",
        nullable=False,
        index=True,
    )

    task_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    step_index: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    state_summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    resolution: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    chroma_synced: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    chroma_synced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONType,
        default=dict,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    task: Mapped[TaskRun | None] = relationship(
        back_populates="experiences",
    )

    __table_args__ = (
        Index(
            "ix_memory_experiences_action_error_code",
            "action",
            "error_code",
        ),
        Index(
            "ix_memory_experiences_outcome_action",
            "outcome",
            "action",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<MemoryExperience id={self.id} action={self.action} "
            f"outcome={self.outcome} error_code={self.error_code}>"
        )
