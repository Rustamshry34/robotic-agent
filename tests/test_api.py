"""
Tests for FastAPI endpoints.

This module tests:
- GET /health - Service health check
- GET /health/db - Database connectivity check
- POST /tasks/run - Task execution
- POST /evaluation/run - Baseline comparison experiment
- Request validation
- Error handling
- Response schemas

All tests use httpx.AsyncClient with ASGITransport for async testing.
Database and executor dependencies are mocked or use simulated backend.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.agent.executor import ExecutionResult, RobotExecutor, SimulatedMCPToolInvoker
from app.core.config import Environment, Settings, get_settings
from app.main import app
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
        agent_enable_memory=False,  # Disable memory for faster tests
        agent_enable_verification=True,
        agent_max_plan_steps=12,
        evaluation_baseline="full_system",
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
def executor(test_settings):
    """Create a test executor."""
    return RobotExecutor(settings=test_settings)


@pytest.fixture
async def client(executor, tool_invoker):
    """
    Create an async HTTP client for testing.

    This fixture:
    1. Sets up app.state with executor and tool_invoker
    2. Creates AsyncClient with ASGITransport
    3. Yields the client
    4. Cleans up app.state after test
    """
    # Set up app state
    app.state.executor = executor
    app.state.tool_invoker = tool_invoker

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_executor():
    """Create a mock executor for isolated endpoint tests."""
    mock = MagicMock(spec=RobotExecutor)
    mock.execute_task = AsyncMock()
    return mock


@pytest.fixture
async def mock_client(mock_executor):
    """Create a client with mocked executor."""
    app.state.executor = mock_executor
    app.state.tool_invoker = MagicMock()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_execution_result():
    """Create a sample ExecutionResult for mocking."""
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return ExecutionResult(
        task_id="test-task-123",
        task_name="test_task",
        success=True,
        status="completed",
        error_code=None,
        error_message=None,
        current_state={"robot_location": "home"},
        steps=[],
        recovery_attempts=0,
        retry_count=0,
        metrics={
            "task_duration_ms": 1000.0,
            "executed_steps": 5.0,
            "completed_steps": 5.0,
            "failed_steps": 0.0,
        },
        started_at=now,
        finished_at=now,
        duration_ms=1000,
    )


@pytest.fixture
def failed_execution_result():
    """Create a failed ExecutionResult for mocking."""
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return ExecutionResult(
        task_id="test-task-456",
        task_name="failed_task",
        success=False,
        status="failed",
        error_code="STEP_FAILED",
        error_message="Step execution failed",
        current_state={},
        steps=[],
        recovery_attempts=2,
        retry_count=3,
        metrics={
            "task_duration_ms": 500.0,
            "executed_steps": 2.0,
            "completed_steps": 1.0,
            "failed_steps": 1.0,
        },
        started_at=now,
        finished_at=now,
        duration_ms=500,
    )


# ----------------------------------------------------------------------
# TestHealthEndpoints
# ----------------------------------------------------------------------


class TestHealthEndpoints:
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test GET /health returns correct response."""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "ok"
        assert "app_name" in data
        assert "version" in data
        assert "environment" in data
        assert data["environment"] == "test"

    @pytest.mark.asyncio
    async def test_health_check_response_schema(self, client):
        """Test health check response has all required fields."""
        response = await client.get("/health")
        data = response.json()

        required_fields = ["status", "app_name", "version", "environment"]
        for field in required_fields:
            assert field in data

    @pytest.mark.asyncio
    async def test_database_health_check_unavailable(self, client):
        """Test GET /health/db when database is unavailable."""
        # In test environment without real DB, this should return degraded status
        response = await client.get("/health/db")

        assert response.status_code == 200
        data = response.json()

        # Should have status and database_connected fields
        assert "status" in data
        assert "database_connected" in data
        assert data["status"] in ["ok", "degraded"]
        assert isinstance(data["database_connected"], bool)


# ----------------------------------------------------------------------
# TestTaskEndpoints
# ----------------------------------------------------------------------


class TestTaskEndpoints:
    @pytest.mark.asyncio
    async def test_run_task_success(self, mock_client, mock_executor, sample_execution_result):
        """Test POST /tasks/run with successful execution."""
        mock_executor.execute_task.return_value = sample_execution_result

        response = await mock_client.post(
            "/tasks/run",
            json={
                "instruction": "Take the red cup on the table",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["status"] == "completed"
        assert data["task_id"] == "test-task-123"
        assert data["task_name"] == "test_task"
        assert "metrics" in data
        assert "duration_ms" in data

    @pytest.mark.asyncio
    async def test_run_task_failure(self, mock_client, mock_executor, failed_execution_result):
        """Test POST /tasks/run with failed execution."""
        mock_executor.execute_task.return_value = failed_execution_result

        response = await mock_client.post(
            "/tasks/run",
            json={
                "instruction": "Take the blue bottle",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is False
        assert data["status"] == "failed"
        assert data["error_code"] == "STEP_FAILED"
        assert data["error_message"] == "Step execution failed"
        assert data["recovery_attempts"] == 2
        assert data["retry_count"] == 3

    @pytest.mark.asyncio
    async def test_run_task_with_custom_name(self, mock_client, mock_executor, sample_execution_result):
        """Test POST /tasks/run with custom task name."""
        mock_executor.execute_task.return_value = sample_execution_result

        response = await mock_client.post(
            "/tasks/run",
            json={
                "instruction": "Take the red cup",
                "task_name": "custom_task_name",
            },
        )

        assert response.status_code == 200

        # Verify executor was called with correct parameters
        mock_executor.execute_task.assert_called_once()
        call_kwargs = mock_executor.execute_task.call_args.kwargs
        assert call_kwargs["task_name"] == "custom_task_name"

    @pytest.mark.asyncio
    async def test_run_task_with_baseline_override(self, mock_client, mock_executor, sample_execution_result):
        """Test POST /tasks/run with baseline override."""
        mock_executor.execute_task.return_value = sample_execution_result

        response = await mock_client.post(
            "/tasks/run",
            json={
                "instruction": "Take the red cup",
                "baseline": "baseline_a",
            },
        )

        assert response.status_code == 200

        call_kwargs = mock_executor.execute_task.call_args.kwargs
        assert call_kwargs["baseline"] == "baseline_a"

    @pytest.mark.asyncio
    async def test_run_task_with_metadata(self, mock_client, mock_executor, sample_execution_result):
        """Test POST /tasks/run with metadata."""
        mock_executor.execute_task.return_value = sample_execution_result

        response = await mock_client.post(
            "/tasks/run",
            json={
                "instruction": "Take the red cup",
                "metadata": {"experiment_id": "exp-123", "user": "test_user"},
            },
        )

        assert response.status_code == 200

        call_kwargs = mock_executor.execute_task.call_args.kwargs
        assert call_kwargs["metadata"]["experiment_id"] == "exp-123"

    @pytest.mark.asyncio
    async def test_run_task_with_real_executor(self, client, backend):
        """Test POST /tasks/run with real executor and simulated backend."""
        response = await client.post(
            "/tasks/run",
            json={
                "instruction": "Take the red cup on the table and put it on the kitchen counter",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["status"] == "completed"
        assert len(data["metrics"]) > 0

        # Verify backend state changed
        state = backend.get_state()
        assert state["data"]["objects"]["red_cup"]["location"] == "kitchen_counter"

    @pytest.mark.asyncio
    async def test_run_task_executor_exception(self, mock_client, mock_executor):
        """Test POST /tasks/run when executor raises exception."""
        mock_executor.execute_task.side_effect = RuntimeError("Executor failed")

        response = await mock_client.post(
            "/tasks/run",
            json={
                "instruction": "Take the red cup",
            },
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


# ----------------------------------------------------------------------
# TestEvaluationEndpoints
# ----------------------------------------------------------------------


class TestEvaluationEndpoints:
    @pytest.mark.asyncio
    async def test_run_evaluation_success(self, mock_client, mock_executor, sample_execution_result):
        """Test POST /evaluation/run with successful evaluations."""
        mock_executor.execute_task.return_value = sample_execution_result

        response = await mock_client.post(
            "/evaluation/run",
            json={
                "instruction": "Take the red cup",
                "baselines": ["baseline_a", "baseline_b"],
                "iterations": 2,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "experiment_id" in data
        assert data["instruction"] == "Take the red cup"
        assert len(data["results"]) == 4 # 2 baselines * 2 iterations
        assert "summaries" in data

    @pytest.mark.asyncio
    async def test_run_evaluation_with_all_baselines(self, mock_client, mock_executor, sample_execution_result):
        """Test POST /evaluation/run with all baselines."""
        mock_executor.execute_task.return_value = sample_execution_result

        response = await mock_client.post(
            "/evaluation/run",
            json={
                "instruction": "Take the red cup",
                "baselines": [
                    "baseline_a",
                    "baseline_b",
                    "baseline_c",
                    "full_system",
                ],
                "iterations": 1,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["results"]) == 4

        # Check all baselines are present
        baselines = {r["baseline"] for r in data["results"]}
        assert baselines == {"baseline_a", "baseline_b", "baseline_c", "full_system"}

    @pytest.mark.asyncio
    async def test_run_evaluation_summary_statistics(self, mock_client, mock_executor):
        """Test that evaluation summary contains correct statistics."""
        # Mock different results for different calls
        from datetime import UTC, datetime

        now = datetime.now(UTC)

        success_result = ExecutionResult(
            task_id="task-1",
            task_name="test",
            success=True,
            status="completed",
            recovery_attempts=0,
            retry_count=0,
            metrics={},
            started_at=now,
            finished_at=now,
            duration_ms=1000,
        )

        failure_result = ExecutionResult(
            task_id="task-2",
            task_name="test",
            success=False,
            status="failed",
            error_code="FAILED",
            recovery_attempts=1,
            retry_count=2,
            metrics={},
            started_at=now,
            finished_at=now,
            duration_ms=500,
        )

        # Alternate between success and failure
        mock_executor.execute_task.side_effect = [
            success_result,
            failure_result,
        ]

        response = await mock_client.post(
            "/evaluation/run",
            json={
                "instruction": "Take the red cup",
                "baselines": ["baseline_a"],
                "iterations": 2,
            },
        )

        assert response.status_code == 200
        data = response.json()

        summaries = data["summaries"]
        assert "baseline_a" in summaries

        baseline_summary = summaries["baseline_a"]
        assert baseline_summary["total_runs"] == 2
        assert baseline_summary["success_count"] == 1
        assert baseline_summary["failure_count"] == 1
        assert baseline_summary["success_rate"] == 0.5
        assert "average_duration_ms" in baseline_summary
        assert baseline_summary["total_recovery_attempts"] == 1
        assert baseline_summary["total_retries"] == 2

    @pytest.mark.asyncio
    async def test_run_evaluation_with_metadata(self, mock_client, mock_executor, sample_execution_result):
        """Test POST /evaluation/run with metadata."""
        mock_executor.execute_task.return_value = sample_execution_result

        response = await mock_client.post(
            "/evaluation/run",
            json={
                "instruction": "Take the red cup",
                "baselines": ["baseline_a"],
                "iterations": 1,
                "metadata": {"experiment_name": "test_experiment"},
            },
        )

        assert response.status_code == 200

        # Verify executor was called with metadata
        # mock_executor.execute_task çağrıldığını doğrula
        assert mock_executor.execute_task.called
        call_kwargs = mock_executor.execute_task.call_args.kwargs
        assert call_kwargs["metadata"]["experiment_name"] == "test_experiment"
        assert "experiment_id" in call_kwargs["metadata"]

    @pytest.mark.asyncio
    async def test_run_evaluation_handles_exceptions(self, mock_client, mock_executor):
        """Test that evaluation continues even when individual runs fail."""
        # First call succeeds, second raises exception
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        success_result = ExecutionResult(
            task_id="task-1",
            task_name="test",
            success=True,
            status="completed",
            started_at=now,
            finished_at=now,
            duration_ms=1000,
        )

        mock_executor.execute_task.side_effect = [
            success_result,
            RuntimeError("Executor failed"),
        ]

        response = await mock_client.post(
            "/evaluation/run",
            json={
                "instruction": "Take the red cup",
                "baselines": ["baseline_a"],
                "iterations": 2,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should have 2 results even though one failed
        assert len(data["results"]) == 2

        # İlk sonuç başarılı
        assert data["results"][0]["success"] is True

        # Second result should indicate error
        error_result = data["results"][1]
        assert error_result["success"] is False
        assert error_result["status"] == "error"
        assert error_result["error_code"] == "BASELINE_EXCEPTION"


# ----------------------------------------------------------------------
# TestRequestValidation
# ----------------------------------------------------------------------


class TestRequestValidation:
    @pytest.mark.asyncio
    async def test_run_task_empty_instruction(self, mock_client):
        """Test POST /tasks/run with empty instruction."""
        response = await mock_client.post(
            "/tasks/run",
            json={
                "instruction": "",
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_run_task_missing_instruction(self, mock_client):
        """Test POST /tasks/run with missing instruction field."""
        response = await mock_client.post(
            "/tasks/run",
            json={},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_run_task_instruction_too_long(self, mock_client):
        """Test POST /tasks/run with instruction exceeding max length."""
        response = await mock_client.post(
            "/tasks/run",
            json={
                "instruction": "x" * 2001,  # Max is 2000
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_run_evaluation_empty_baselines(self, mock_client):
        """Test POST /evaluation/run with empty baselines list."""
        response = await mock_client.post(
            "/evaluation/run",
            json={
                "instruction": "Take the red cup",
                "baselines": [],
            },
        )

        # Should still work, just no results
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "At least one baseline" in data["detail"]
        # assert len(data["results"]) == 0

    @pytest.mark.asyncio
    async def test_run_evaluation_invalid_iterations(self, mock_client):
        """Test POST /evaluation/run with invalid iterations."""
        response = await mock_client.post(
            "/evaluation/run",
            json={
                "instruction": "Take the red cup",
                "baselines": ["baseline_a"],
                "iterations": 0,  # Must be >= 1
            },
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_run_evaluation_iterations_too_high(self, mock_client):
        """Test POST /evaluation/run with iterations exceeding max."""
        response = await mock_client.post(
            "/evaluation/run",
            json={
                "instruction": "Take the red cup",
                "baselines": ["baseline_a"],
                "iterations": 51,  # Max is 50
            },
        )

        assert response.status_code == 422


# ----------------------------------------------------------------------
# TestResponseSchemas
# ----------------------------------------------------------------------


class TestResponseSchemas:
    @pytest.mark.asyncio
    async def test_task_response_schema(self, mock_client, mock_executor, sample_execution_result):
        """Test that task response matches expected schema."""
        mock_executor.execute_task.return_value = sample_execution_result

        response = await mock_client.post(
            "/tasks/run",
            json={
                "instruction": "Take the red cup",
            },
        )

        data = response.json()

        # Check all expected fields
        expected_fields = [
            "task_id",
            "task_name",
            "success",
            "status",
            "error_code",
            "error_message",
            "recovery_attempts",
            "retry_count",
            "metrics",
            "duration_ms",
        ]

        for field in expected_fields:
            assert field in data

    @pytest.mark.asyncio
    async def test_evaluation_response_schema(self, mock_client, mock_executor, sample_execution_result):
        """Test that evaluation response matches expected schema."""
        mock_executor.execute_task.return_value = sample_execution_result

        response = await mock_client.post(
            "/evaluation/run",
            json={
                "instruction": "Take the red cup",
                "baselines": ["baseline_a"],
                "iterations": 1,
            },
        )

        data = response.json()

        # Check top-level fields
        assert "experiment_id" in data
        assert "instruction" in data
        assert "results" in data
        assert "summaries" in data

        # Check result schema
        if data["results"]:
            result = data["results"][0]
            result_fields = [
                "baseline",
                "iteration",
                "success",
                "status",
                "error_code",
                "recovery_attempts",
                "retry_count",
                "metrics",
                "duration_ms",
            ]
            for field in result_fields:
                assert field in result


# ----------------------------------------------------------------------
# TestErrorHandling
# ----------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_invalid_json_body(self, mock_client):
        """Test handling of invalid JSON body."""
        response = await mock_client.post(
            "/tasks/run",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_wrong_content_type(self, mock_client):
        """Test handling of wrong content type."""
        response = await mock_client.post(
            "/tasks/run",
            content="instruction=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        # FastAPI should reject this
        assert response.status_code in [422, 415]


# ----------------------------------------------------------------------
# TestIntegration
# ----------------------------------------------------------------------


class TestIntegration:
    @pytest.mark.asyncio
    async def test_complete_workflow(self, client, backend):
        """Test complete workflow: health check -> task execution -> verify state."""
        # 1. Health check
        health_response = await client.get("/health")
        assert health_response.status_code == 200

        # 2. Execute task
        task_response = await client.post(
            "/tasks/run",
            json={
                "instruction": "Take the red cup on the table and put it on the kitchen counter",
            },
        )
        assert task_response.status_code == 200
        task_data = task_response.json()
        assert task_data["success"] is True

        # 3. Verify backend state
        state = backend.get_state()
        assert state["data"]["objects"]["red_cup"]["location"] == "kitchen_counter"
        assert state["data"]["robot"]["holding_object"] is None

    @pytest.mark.asyncio
    async def test_multiple_tasks_sequential(self, client, backend):
        """Test executing multiple tasks sequentially."""
        # First task: move red cup
        response1 = await client.post(
            "/tasks/run",
            json={
                "instruction": "Take the red cup to the kitchen counter",
            },
        )
        assert response1.status_code == 200
        assert response1.json()["success"] is True

        # Reset backend for second task
        backend.reset_scene()

        # Second task: verify initial state
        response2 = await client.post(
            "/tasks/run",
            json={
                "instruction": "Check the robot state",
            },
        )
        assert response2.status_code == 200

    @pytest.mark.asyncio
    async def test_evaluation_with_real_executor(self, client):
        """Test evaluation endpoint with real executor."""
        response = await client.post(
            "/evaluation/run",
            json={
                "instruction": "Take the red cup",
                "baselines": ["baseline_a", "baseline_c"],
                "iterations": 1,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["results"]) == 2
        assert "summaries" in data
