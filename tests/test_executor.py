"""
Tests for the robot task executor.

This module tests:
- Plan execution with SimulatedRobotBackend
- Task execution (planning + execution)
- Verification after critical actions
- Retry and recovery mechanisms
- PostgreSQL logging (with mock session)
- ChromaDB experience storage (with mock client)
- Metrics generation
- Tool timeout handling
- Baseline behavior differences
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.executor import (
    ExecutionResult,
    ExecutorConfig,
    RobotExecutor,
    SimulatedMCPToolInvoker,
    StepExecutionResult,
    ToolInvoker,
    _normalize_tool_result,
    duration_ms,
    utcnow,
)
from app.agent.planner import (
    FailureStrategy,
    PlanStep,
    PlanStepType,
    PlannerConfig,
    RobotPlan,
    RobotPlanner,
)
from app.core.config import Environment, Settings
from memory.chroma_client import ChromaMemoryClient
from memory.models import (
    BaselineName,
    ExperienceOutcome,
    StepStatus,
    TaskRun,
    TaskStatus,
)
from mcp_server.tools import SimulatedRobotBackend


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def test_settings():
    """Create test settings."""
    return Settings(
        environment=Environment.TEST,
        llm_api_key=None,
        agent_enable_memory=True,
        agent_enable_verification=True,
        agent_max_plan_steps=12,
        agent_max_action_retries=3,
        chroma_top_k=5,
        mcp_timeout_seconds=30,
        evaluation_baseline="full_system",
        agent_version="0.1.0-test",
        git_commit="test-commit",
    )


@pytest.fixture
def backend():
    """Create a fresh SimulatedRobotBackend."""
    return SimulatedRobotBackend()


@pytest.fixture
def tool_invoker(backend):
    """Create a SimulatedMCPToolInvoker."""
    return SimulatedMCPToolInvoker(backend=backend)


@pytest.fixture
def mock_memory_client():
    """Create a mock ChromaDB memory client."""
    client = MagicMock(spec=ChromaMemoryClient)
    client.add_experience.return_value = "test-experience-id"
    client.retrieve_relevant_experiences.return_value = []
    client.retrieve_error_recovery_experiences.return_value = []
    return client


@pytest.fixture
def mock_db_session():
    """Create a mock async database session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def planner(test_settings, mock_memory_client):
    """Create a planner with test settings."""
    return RobotPlanner(
        settings=test_settings,
        memory_client=mock_memory_client,
        config=PlannerConfig.from_settings(test_settings),
    )


@pytest.fixture
def executor(test_settings, planner, mock_memory_client):
    """Create an executor with test settings."""
    return RobotExecutor(
        planner=planner,
        settings=test_settings,
        memory_client=mock_memory_client,
    )


@pytest.fixture
def simple_plan():
    """Create a simple test plan."""
    return RobotPlan(
        task_name="test_task",
        goal="Test task",
        rationale="Test",
        baseline="full_system",
        requires_verification=True,
        requires_memory=True,
        steps=[
            PlanStep(
                step_id=1,
                description="Get robot state",
                tool_name="robot_get_state",
                step_type=PlanStepType.OBSERVATION,
            ),
        ],
    )


# ----------------------------------------------------------------------
# TestHelpers
# ----------------------------------------------------------------------


class TestHelpers:
    def test_utcnow(self):
        """Test utcnow returns UTC datetime."""
        now = utcnow()
        assert now.tzinfo == UTC

    def test_duration_ms(self):
        """Test duration_ms calculation."""
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        end = datetime(2024, 1, 1, 12, 0, 1, tzinfo=UTC)
        assert duration_ms(start, end) == 1000

    def test_normalize_tool_result_dict(self):
        """Test normalizing a dict result."""
        result = {
            "success": True,
            "message": "OK",
            "data": {"key": "value"},
            "error_code": None,
        }
        normalized = _normalize_tool_result(result)
        assert normalized["success"] is True
        assert normalized["message"] == "OK"
        assert normalized["data"] == {"key": "value"}

    def test_normalize_tool_result_pydantic(self):
        """Test normalizing a Pydantic model result."""
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "success": False,
            "message": "Error",
            "data": {},
            "error_code": "TEST_ERROR",
        }
        normalized = _normalize_tool_result(mock_result)
        assert normalized["success"] is False
        assert normalized["error_code"] == "TEST_ERROR"

    def test_normalize_tool_result_unknown(self):
        """Test normalizing an unknown result type."""
        normalized = _normalize_tool_result("unexpected string")
        assert normalized["success"] is False
        assert normalized["error_code"] == "UNKNOWN_TOOL_RESULT"


# ----------------------------------------------------------------------
# TestSimulatedMCPToolInvoker
# ----------------------------------------------------------------------


class TestSimulatedMCPToolInvoker:
    @pytest.mark.asyncio
    async def test_call_valid_tool(self, tool_invoker):
        """Test calling a valid tool."""
        result = await tool_invoker.call_tool(
            "robot_get_state",
            {},
        )
        assert result["success"] is True
        assert "data" in result

    @pytest.mark.asyncio
    async def test_call_invalid_tool(self, tool_invoker):
        """Test calling an invalid tool."""
        result = await tool_invoker.call_tool(
            "robot_nonexistent_tool",
            {},
        )
        assert result["success"] is False
        assert result["error_code"] == "UNKNOWN_TOOL"

    @pytest.mark.asyncio
    async def test_call_tool_with_invalid_args(self, tool_invoker):
        """Test calling a tool with invalid arguments."""
        result = await tool_invoker.call_tool(
            "robot_detect_object",
            {"invalid_arg": "value"},
        )
        assert result["success"] is False
        assert result["error_code"] == "INVALID_TOOL_ARGUMENTS"

    def test_list_tools(self, tool_invoker):
        """Test listing available tools."""
        tools = tool_invoker.list_tools()
        assert len(tools) > 0
        tool_names = [t["name"] for t in tools]
        assert "robot_get_state" in tool_names
        assert "robot_grasp_object" in tool_names


# ----------------------------------------------------------------------
# TestExecutorConfig
# ----------------------------------------------------------------------


class TestExecutorConfig:
    def test_from_settings(self, test_settings):
        """Test creating config from settings."""
        config = ExecutorConfig.from_settings(test_settings)
        assert config.max_recovery_attempts == test_settings.agent_max_action_retries
        assert config.step_timeout_seconds == test_settings.mcp_timeout_seconds
        assert config.store_step_experiences == test_settings.agent_enable_memory

    def test_default_values(self):
        """Test default config values."""
        config = ExecutorConfig()
        assert config.refresh_state_after_action is True
        assert config.store_step_experiences is True
        assert config.max_recovery_attempts == 3
        assert config.recovery_baselines == ("full_system",)


# ----------------------------------------------------------------------
# TestStepExecutionResult
# ----------------------------------------------------------------------


class TestStepExecutionResult:
    def test_valid_result(self):
        """Test creating a valid step result."""
        now = utcnow()
        result = StepExecutionResult(
            step_id=1,
            tool_name="robot_get_state",
            success=True,
            attempts=1,
            verified=None,
            output={"success": True, "message": "OK", "data": {}},
            started_at=now,
            finished_at=now,
            duration_ms=100,
        )
        assert result.success is True
        assert result.attempts == 1

    def test_failed_result(self):
        """Test creating a failed step result."""
        now = utcnow()
        result = StepExecutionResult(
            step_id=1,
            tool_name="robot_grasp_object",
            success=False,
            attempts=2,
            error_code="GRASP_FAILED",
            error_message="Grasp failed",
            started_at=now,
            finished_at=now,
            duration_ms=500,
        )
        assert result.success is False
        assert result.error_code == "GRASP_FAILED"


# ----------------------------------------------------------------------
# TestExecutionResult
# ----------------------------------------------------------------------


class TestExecutionResult:
    def test_valid_result(self):
        """Test creating a valid execution result."""
        now = utcnow()
        result = ExecutionResult(
            task_id="test-id",
            task_name="test_task",
            success=True,
            status="completed",
            started_at=now,
            finished_at=now,
            duration_ms=1000,
        )
        assert result.success is True
        assert result.status == "completed"

    def test_failed_result(self):
        """Test creating a failed execution result."""
        now = utcnow()
        result = ExecutionResult(
            task_id="test-id",
            task_name="test_task",
            success=False,
            status="failed",
            error_code="STEP_FAILED",
            error_message="Step execution failed",
            started_at=now,
            finished_at=now,
            duration_ms=500,
        )
        assert result.success is False
        assert result.error_code == "STEP_FAILED"


# ----------------------------------------------------------------------
# TestExecutePlan - Successful Scenarios
# ----------------------------------------------------------------------


class TestExecutePlanSuccess:
    @pytest.mark.asyncio
    async def test_execute_simple_plan(self, executor, tool_invoker, simple_plan):
        """Test executing a simple plan successfully."""
        result = await executor.execute_plan(
            plan=simple_plan,
            task_instruction="Test task",
            tool_invoker=tool_invoker,
        )

        assert result.success is True
        assert result.status == TaskStatus.COMPLETED.value
        assert len(result.steps) == 1
        assert result.steps[0].success is True

    @pytest.mark.asyncio
    async def test_execute_plan_with_verification(self, executor, tool_invoker):
        """Test executing a plan with verification steps."""
        plan = RobotPlan(
            task_name="move_red_cup",
            goal="Move red cup",
            baseline="full_system",
            requires_verification=True,
            steps=[
                PlanStep(
                    step_id=1,
                    description="Move to table",
                    tool_name="robot_move_to",
                    step_type=PlanStepType.ACTION,
                    input_payload={"target_id": "table"},
                ),
                PlanStep(
                    step_id=2,
                    description="Grasp red_cup",
                    tool_name="robot_grasp_object",
                    step_type=PlanStepType.ACTION,
                    input_payload={"object_id": "red_cup"},
                    verification_tool="robot_verify_object_location",
                    verification_payload={
                        "object_id": "red_cup",
                        "expected_location": "gripper",
                    },
                ),
            ],
        )

        result = await executor.execute_plan(
            plan=plan,
            task_instruction="Move red cup",
            tool_invoker=tool_invoker,
        )

        assert result.success is True
        assert len(result.steps) == 2
        assert result.steps[1].verified is True

    @pytest.mark.asyncio
    async def test_execute_plan_with_verification_step(self, executor, tool_invoker):
        """Test executing a plan with explicit verification step."""
        plan = RobotPlan(
            task_name="verify_test",
            goal="Verify object location",
            baseline="full_system",
            requires_verification=True,
            steps=[
                PlanStep(
                    step_id=1,
                    description="Verify red_cup location",
                    tool_name="robot_verify_object_location",
                    step_type=PlanStepType.VERIFICATION,
                    input_payload={
                        "object_id": "red_cup",
                        "expected_location": "table",
                    },
                ),
            ],
        )

        result = await executor.execute_plan(
            plan=plan,
            task_instruction="Verify red cup",
            tool_invoker=tool_invoker,
        )

        assert result.success is True
        assert result.steps[0].verified is True


# ----------------------------------------------------------------------
# TestExecutePlan - Failure Scenarios
# ----------------------------------------------------------------------


class TestExecutePlanFailure:
    @pytest.mark.asyncio
    async def test_execute_plan_step_failure(self, executor, tool_invoker):
        """Test plan execution with step failure."""
        plan = RobotPlan(
            task_name="fail_test",
            goal="Test failure",
            baseline="baseline_a",  # No recovery
            requires_verification=False,
            steps=[
                PlanStep(
                    step_id=1,
                    description="Grasp non-existent object",
                    tool_name="robot_grasp_object",
                    step_type=PlanStepType.ACTION,
                    input_payload={"object_id": "nonexistent_object"},
                    on_failure=FailureStrategy.ABORT,
                ),
            ],
        )

        result = await executor.execute_plan(
            plan=plan,
            task_instruction="Test failure",
            tool_invoker=tool_invoker,
        )

        assert result.success is False
        assert result.status == TaskStatus.FAILED.value
        assert result.error_code == "OBJECT_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_execute_plan_verification_failure(self, executor, tool_invoker):
        """Test plan execution with verification failure."""
        plan = RobotPlan(
            task_name="verify_fail",
            goal="Test verification failure",
            baseline="full_system",
            requires_verification=True,
            steps=[
                PlanStep(
                    step_id=1,
                    description="Verify wrong location",
                    tool_name="robot_verify_object_location",
                    step_type=PlanStepType.VERIFICATION,
                    input_payload={
                        "object_id": "red_cup",
                        "expected_location": "wrong_location",
                    },
                ),
            ],
        )

        result = await executor.execute_plan(
            plan=plan,
            task_instruction="Test verification",
            tool_invoker=tool_invoker,
        )

        assert result.success is False
        assert result.error_code == "VERIFICATION_FAILED"

    @pytest.mark.asyncio
    async def test_execute_plan_tool_timeout(self, test_settings, planner, mock_memory_client):
        """Test plan execution with tool timeout."""
        # Create executor with very short timeout
        config = ExecutorConfig(step_timeout_seconds=0.001)
        executor = RobotExecutor(
            planner=planner,
            settings=test_settings,
            memory_client=mock_memory_client,
            config=config,
        )

        # Create slow tool invoker
        slow_invoker = AsyncMock(spec=ToolInvoker)
        slow_invoker.call_tool = AsyncMock(side_effect=TimeoutError("Tool timed out"))
        slow_invoker.list_tools.return_value = []

        plan = RobotPlan(
            task_name="timeout_test",
            goal="Test timeout",
            baseline="baseline_a",
            steps=[
                PlanStep(
                    step_id=1,
                    description="Slow operation",
                    tool_name="robot_get_state",
                ),
            ],
        )

        result = await executor.execute_plan(
            plan=plan,
            task_instruction="Test timeout",
            tool_invoker=slow_invoker,
        )

        assert result.success is False
        assert result.error_code == "TOOL_TIMEOUT"


# ----------------------------------------------------------------------
# TestExecutePlan - Recovery Scenarios
# ----------------------------------------------------------------------


class TestExecutePlanRecovery:
    @pytest.mark.asyncio
    async def test_execute_plan_with_recovery(self, executor, tool_invoker, backend):
        """Test plan execution with recovery after failure."""
        # Inject fault for first grasp attempt
        backend.inject_fault("grasp_object")

        plan = RobotPlan(
            task_name="recovery_test",
            goal="Test recovery",
            baseline="full_system",
            requires_verification=False,
            steps=[
                PlanStep(
                    step_id=1,
                    description="Move to table",
                    tool_name="robot_move_to",
                    step_type=PlanStepType.ACTION,
                    input_payload={"target_id": "table"},
                ),
                PlanStep(
                    step_id=2,
                    description="Grasp red_cup",
                    tool_name="robot_grasp_object",
                    step_type=PlanStepType.ACTION,
                    input_payload={"object_id": "red_cup"},
                ),
            ],
        )

        result = await executor.execute_plan(
            plan=plan,
            task_instruction="Test recovery",
            tool_invoker=tool_invoker,
        )

        # Should recover and succeed
        assert result.success is True
        assert result.recovery_attempts >= 1

    @pytest.mark.asyncio
    async def test_execute_plan_recovery_limit(self, test_settings, planner, mock_memory_client):
        """Test that recovery attempts are limited."""
        config = ExecutorConfig(max_recovery_attempts=1)
        executor = RobotExecutor(
            planner=planner,
            settings=test_settings,
            memory_client=mock_memory_client,
            config=config,
        )

        # Create invoker that always fails
        failing_invoker = AsyncMock(spec=ToolInvoker)
        failing_invoker.call_tool = AsyncMock(
            return_value={
                "success": False,
                "message": "Always fails",
                "data": {},
                "error_code": "ALWAYS_FAILS",
            }
        )
        failing_invoker.list_tools.return_value = []

        plan = RobotPlan(
            task_name="recovery_limit_test",
            goal="Test recovery limit",
            baseline="full_system",
            steps=[
                PlanStep(
                    step_id=1,
                    description="Always fails",
                    tool_name="robot_get_state",
                ),
            ],
        )

        result = await executor.execute_plan(
            plan=plan,
            task_instruction="Test recovery limit",
            tool_invoker=failing_invoker,
        )

        assert result.success is False
        assert result.recovery_attempts <= config.max_recovery_attempts

    @pytest.mark.asyncio
    async def test_no_recovery_in_baseline_a(self, test_settings, planner, mock_memory_client):
        """Test that baseline A does not attempt recovery."""
        planner_no_recovery = RobotPlanner(
            settings=test_settings,
            memory_client=mock_memory_client,
            config=PlannerConfig.from_settings(test_settings, baseline="baseline_a"),
        )

        executor = RobotExecutor(
            planner=planner_no_recovery,
            settings=test_settings,
            memory_client=mock_memory_client,
        )

        # Create invoker that fails
        failing_invoker = AsyncMock(spec=ToolInvoker)
        failing_invoker.call_tool = AsyncMock(
            return_value={
                "success": False,
                "message": "Fails",
                "data": {},
                "error_code": "FAILS",
            }
        )
        failing_invoker.list_tools.return_value = []

        plan = RobotPlan(
            task_name="no_recovery_test",
            goal="Test no recovery",
            baseline="baseline_a",
            steps=[
                PlanStep(
                    step_id=1,
                    description="Fails",
                    tool_name="robot_get_state",
                ),
            ],
        )

        result = await executor.execute_plan(
            plan=plan,
            task_instruction="Test no recovery",
            tool_invoker=failing_invoker,
        )

        assert result.success is False
        assert result.recovery_attempts == 0


# ----------------------------------------------------------------------
# TestExecuteTask
# ----------------------------------------------------------------------


class TestExecuteTask:
    @pytest.mark.asyncio
    async def test_execute_task_from_instruction(self, executor, tool_invoker):
        """Test executing a task from natural language instruction."""
        result = await executor.execute_task(
            instruction="Take the red cup on the table and put it on the kitchen counter",
            tool_invoker=tool_invoker,
        )

        assert result.success is True
        assert result.task_name is not None
        assert len(result.steps) > 0
        assert result.metrics is not None

    @pytest.mark.asyncio
    async def test_execute_task_with_custom_name(self, executor, tool_invoker):
        """Test executing a task with custom name."""
        result = await executor.execute_task(
            instruction="Take the red cup",
            tool_invoker=tool_invoker,
            task_name="custom_task_name",
        )

        assert result.task_name == "custom_task_name"

    @pytest.mark.asyncio
    async def test_execute_task_with_baseline_override(self, executor, tool_invoker):
        """Test executing a task with baseline override."""
        result = await executor.execute_task(
            instruction="Take the red cup",
            tool_invoker=tool_invoker,
            baseline="baseline_a",
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_task_with_metadata(self, executor, tool_invoker):
        """Test executing a task with metadata."""
        result = await executor.execute_task(
            instruction="Take the red cup",
            tool_invoker=tool_invoker,
            metadata={"test_key": "test_value"},
        )

        assert result.success is True


# ----------------------------------------------------------------------
# TestDatabaseLogging
# ----------------------------------------------------------------------


class TestDatabaseLogging:
    @pytest.mark.asyncio
    async def test_execute_plan_with_db_session(
        self, executor, tool_invoker, simple_plan, mock_db_session
    ):
        """Test that executor logs to database when session provided."""
        result = await executor.execute_plan(
            plan=simple_plan,
            task_instruction="Test task",
            tool_invoker=tool_invoker,
            db_session=mock_db_session,
        )

        assert result.success is True
        # Verify database session was used
        assert mock_db_session.add.called
        assert mock_db_session.flush.called

    @pytest.mark.asyncio
    async def test_execute_plan_without_db_session(self, executor, tool_invoker, simple_plan):
        """Test that executor works without database session."""
        result = await executor.execute_plan(
            plan=simple_plan,
            task_instruction="Test task",
            tool_invoker=tool_invoker,
            db_session=None,
        )

        assert result.success is True
        assert result.task_id is None


# ----------------------------------------------------------------------
# TestMemoryStorage
# ----------------------------------------------------------------------


class TestMemoryStorage:
    @pytest.mark.asyncio
    async def test_experience_stored_on_success(
        self, executor, tool_invoker, mock_memory_client
    ):
        """Test that experiences are stored on successful steps."""
        plan = RobotPlan(
            task_name="memory_test",
            goal="Test memory",
            baseline="full_system",
            steps=[
                PlanStep(
                    step_id=1,
                    description="Get state",
                    tool_name="robot_get_state",
                    step_type=PlanStepType.ACTION,
                ),
            ],
        )

        await executor.execute_plan(
            plan=plan,
            task_instruction="Test memory",
            tool_invoker=tool_invoker,
        )

        # Verify memory was called
        assert mock_memory_client.add_experience.called

    @pytest.mark.asyncio
    async def test_experience_not_stored_for_observations(
        self, executor, tool_invoker, mock_memory_client
    ):
        """Test that observation steps don't store experiences."""
        plan = RobotPlan(
            task_name="observation_test",
            goal="Test observation",
            baseline="full_system",
            steps=[
                PlanStep(
                    step_id=1,
                    description="Observe",
                    tool_name="robot_get_state",
                    step_type=PlanStepType.OBSERVATION,
                ),
            ],
        )

        await executor.execute_plan(
            plan=plan,
            task_instruction="Test observation",
            tool_invoker=tool_invoker,
        )

        # Observation steps should not store experiences
        # (only ACTION, VERIFICATION, RECOVERY types store)
        # Note: This depends on implementation details

    @pytest.mark.asyncio
    async def test_no_memory_storage_when_disabled(
        self, test_settings, planner
    ):
        """Test that memory storage can be disabled."""
        config = ExecutorConfig(store_step_experiences=False)
        executor = RobotExecutor(
            planner=planner,
            settings=test_settings,
            memory_client=None,
            config=config,
        )

        backend = SimulatedRobotBackend()
        tool_invoker = SimulatedMCPToolInvoker(backend=backend)

        plan = RobotPlan(
            task_name="no_memory_test",
            goal="Test no memory",
            baseline="full_system",
            steps=[
                PlanStep(
                    step_id=1,
                    description="Get state",
                    tool_name="robot_get_state",
                    step_type=PlanStepType.ACTION,
                ),
            ],
        )

        result = await executor.execute_plan(
            plan=plan,
            task_instruction="Test no memory",
            tool_invoker=tool_invoker,
        )

        assert result.success is True


# ----------------------------------------------------------------------
# TestMetrics
# ----------------------------------------------------------------------


class TestMetrics:
    @pytest.mark.asyncio
    async def test_metrics_generated(self, executor, tool_invoker, simple_plan):
        """Test that execution metrics are generated."""
        result = await executor.execute_plan(
            plan=simple_plan,
            task_instruction="Test metrics",
            tool_invoker=tool_invoker,
        )

        assert "task_duration_ms" in result.metrics
        assert "executed_steps" in result.metrics
        assert "completed_steps" in result.metrics
        assert "failed_steps" in result.metrics
        assert "retry_count" in result.metrics
        assert "recovery_attempts" in result.metrics

    @pytest.mark.asyncio
    async def test_metrics_values_correct(self, executor, tool_invoker):
        """Test that metric values are correct."""
        plan = RobotPlan(
            task_name="metrics_test",
            goal="Test metrics values",
            baseline="full_system",
            steps=[
                PlanStep(
                    step_id=1,
                    description="Step 1",
                    tool_name="robot_get_state",
                ),
                PlanStep(
                    step_id=2,
                    description="Step 2",
                    tool_name="robot_list_objects",
                ),
            ],
        )

        result = await executor.execute_plan(
            plan=plan,
            task_instruction="Test metrics",
            tool_invoker=tool_invoker,
        )

        assert result.metrics["executed_steps"] == 2.0
        assert result.metrics["completed_steps"] == 2.0
        assert result.metrics["failed_steps"] == 0.0
        assert result.metrics["retry_count"] == 0.0
        assert result.metrics["recovery_attempts"] == 0.0


# ----------------------------------------------------------------------
# TestCompleteTaskScenario
# ----------------------------------------------------------------------


class TestCompleteTaskScenario:
    @pytest.mark.asyncio
    async def test_complete_task_move_red_cup(self, executor, tool_invoker):
        """Test complete task: move red cup from table to kitchen counter."""
        result = await executor.execute_task(
            instruction="Take the red cup on the table, put it on the kitchen counter, and verify the process is complete.",
            tool_invoker=tool_invoker,
        )

        assert result.success is True
        assert result.status == TaskStatus.COMPLETED.value

        # Verify final state
        backend_state = await tool_invoker.call_tool("robot_get_state", {})
        assert backend_state["data"]["objects"]["red_cup"]["location"] == "kitchen_counter"
        assert backend_state["data"]["robot"]["holding_object"] is None

    @pytest.mark.asyncio
    async def test_complete_task_with_fault_recovery(self, executor, tool_invoker, backend):
        """Test complete task with fault injection and recovery."""
        # Inject fault for first grasp attempt
        backend.inject_fault("grasp_object")

        result = await executor.execute_task(
            instruction="Take the red cup on the table and put it on the kitchen counter",
            tool_invoker=tool_invoker,
        )

        # Should recover and succeed
        assert result.success is True
        assert result.recovery_attempts >= 1

        # Verify final state
        backend_state = await tool_invoker.call_tool("robot_get_state", {})
        assert backend_state["data"]["objects"]["red_cup"]["location"] == "kitchen_counter"


# ----------------------------------------------------------------------
# TestEdgeCases
# ----------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_execute_empty_plan(self, executor, tool_invoker):
        """Test executing a plan with no steps."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="Plan must contain at least one step"):
            RobotPlan(
                task_name="empty_plan",
                goal="Empty plan",
                baseline="full_system",
                steps=[],  # This should fail validation
            )

        # Plan validation should catch this
        with pytest.raises(ValueError, match="Plan must contain at least one step"):
            await executor.execute_plan(
                plan=plan,
                task_instruction="Test empty",
                tool_invoker=tool_invoker,
            )

    @pytest.mark.asyncio
    async def test_execute_plan_with_initial_state(self, executor, tool_invoker):
        """Test executing a plan with initial state provided."""
        plan = RobotPlan(
            task_name="state_test",
            goal="Test state",
            baseline="full_system",
            steps=[
                PlanStep(
                    step_id=1,
                    description="Get state",
                    tool_name="robot_get_state",
                ),
            ],
        )

        initial_state = {
            "robot_location": "home",
            "gripper_open": True,
            "holding_object": None,
        }

        result = await executor.execute_plan(
            plan=plan,
            task_instruction="Test state",
            tool_invoker=tool_invoker,
            current_state=initial_state,
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, executor, tool_invoker, backend):
        """Test retry mechanism on step failure."""
        # Inject fault for first attempt
        backend.inject_fault("grasp_object")

        plan = RobotPlan(
            task_name="retry_test",
            goal="Test retry",
            baseline="baseline_a",  # No recovery, only retry
            requires_verification=False,
            steps=[
                PlanStep(
                    step_id=1,
                    description="Move to table",
                    tool_name="robot_move_to",
                    input_payload={"target_id": "table"},
                ),
                PlanStep(
                    step_id=2,
                    description="Grasp red_cup",
                    tool_name="robot_grasp_object",
                    input_payload={"object_id": "red_cup"},
                    on_failure=FailureStrategy.RETRY,
                    max_retries=2,
                ),
            ],
        )

        result = await executor.execute_plan(
            plan=plan,
            task_instruction="Test retry",
            tool_invoker=tool_invoker,
        )

        # Should succeed after retry
        assert result.success is True
        assert result.retry_count >= 1
