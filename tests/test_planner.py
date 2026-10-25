"""
Tests for the robot task planner.

This module tests:
- PlannerConfig and baseline settings
- PlanStep and RobotPlan validation
- Heuristic planner (LLM-free)
- Recovery plan generation
- Memory retrieval integration
- Helper functions
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.agent.planner import (
    DEFAULT_TOOL_SPECS,
    FailureStrategy,
    PlanStep,
    PlanStepType,
    PlannerConfig,
    RobotPlan,
    RobotPlanner,
    _format_memories,
    _renumber_steps,
    create_planner,
    slugify,
)
from app.core.config import Environment, Settings
from memory.chroma_client import ChromaMemoryClient


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def test_settings():
    """Create test settings without LLM API key."""
    return Settings(
        environment=Environment.TEST,
        llm_api_key=None,
        agent_enable_memory=True,
        agent_enable_verification=True,
        agent_max_plan_steps=12,
        chroma_top_k=5,
        evaluation_baseline="full_system",
    )


@pytest.fixture
def mock_memory_client():
    """Create a mock ChromaDB memory client."""
    client = MagicMock(spec=ChromaMemoryClient)
    client.retrieve_relevant_experiences.return_value = []
    client.retrieve_error_recovery_experiences.return_value = []
    client.retrieve_success_experiences.return_value = []
    return client


@pytest.fixture
def planner(test_settings, mock_memory_client):
    """Create a planner with test settings and mock memory."""
    return create_planner(
        settings=test_settings,
        memory_client=mock_memory_client,
    )


# ----------------------------------------------------------------------
# TestPlannerConfig
# ----------------------------------------------------------------------


class TestPlannerConfig:
    def test_baseline_a_disables_memory_and_verification(self, test_settings):
        """Baseline A should disable both memory and verification."""
        config = PlannerConfig.from_settings(test_settings, baseline="baseline_a")

        assert config.enable_memory is False
        assert config.enable_verification is False
        assert config.baseline == "baseline_a"

    def test_baseline_b_enables_memory_only(self, test_settings):
        """Baseline B should enable memory but disable verification."""
        config = PlannerConfig.from_settings(test_settings, baseline="baseline_b")

        assert config.enable_memory is True
        assert config.enable_verification is False
        assert config.baseline == "baseline_b"

    def test_baseline_c_enables_verification_only(self, test_settings):
        """Baseline C should enable verification but disable memory."""
        config = PlannerConfig.from_settings(test_settings, baseline="baseline_c")

        assert config.enable_memory is False
        assert config.enable_verification is True
        assert config.baseline == "baseline_c"

    def test_full_system_enables_both(self, test_settings):
        """Full system should enable both memory and verification."""
        config = PlannerConfig.from_settings(test_settings, baseline="full_system")

        assert config.enable_memory is True
        assert config.enable_verification is True
        assert config.baseline == "full_system"

    def test_baseline_aliases(self, test_settings):
        """Test baseline name aliases (a, b, c)."""
        config_a = PlannerConfig.from_settings(test_settings, baseline="a")
        assert config_a.baseline == "baseline_a"

        config_b = PlannerConfig.from_settings(test_settings, baseline="b")
        assert config_b.baseline == "baseline_b"

        config_c = PlannerConfig.from_settings(test_settings, baseline="c")
        assert config_c.baseline == "baseline_c"

    def test_respects_settings_flags(self, test_settings):
        """Test that settings flags override baseline defaults."""
        test_settings.agent_enable_memory = False
        test_settings.agent_enable_verification = False

        config = PlannerConfig.from_settings(test_settings, baseline="full_system")

        assert config.enable_memory is False
        assert config.enable_verification is False


# ----------------------------------------------------------------------
# TestPlanStep
# ----------------------------------------------------------------------


class TestPlanStep:
    def test_valid_plan_step(self):
        """Test creating a valid plan step."""
        step = PlanStep(
            step_id=1,
            description="Move to table",
            tool_name="robot_move_to",
            step_type=PlanStepType.ACTION,
            input_payload={"target_id": "table"},
            expected_state="Robot is at table",
        )

        assert step.step_id == 1
        assert step.tool_name == "robot_move_to"
        assert step.step_type == PlanStepType.ACTION

    def test_plan_step_with_verification(self):
        """Test plan step with verification tool."""
        step = PlanStep(
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
        )

        assert step.verification_tool == "robot_verify_object_location"
        assert step.verification_payload["expected_location"] == "gripper"

    def test_plan_step_default_values(self):
        """Test plan step default values."""
        step = PlanStep(
            description="Test step",
            tool_name="robot_get_state",
        )

        assert step.step_id == 0
        assert step.step_type == PlanStepType.ACTION
        assert step.input_payload == {}
        assert step.expected_state == ""
        assert step.verification_tool is None
        assert step.on_failure == FailureStrategy.REPLAN
        assert step.max_retries == 1

    def test_plan_step_validation_empty_description(self):
        """Test that empty description raises validation error."""
        with pytest.raises(ValueError):
            PlanStep(
                description="",
                tool_name="robot_get_state",
            )

    def test_plan_step_validation_empty_tool_name(self):
        """Test that empty tool_name raises validation error."""
        with pytest.raises(ValueError):
            PlanStep(
                description="Test",
                tool_name="",
            )


# ----------------------------------------------------------------------
# TestRobotPlan
# ----------------------------------------------------------------------


class TestRobotPlan:
    def test_valid_robot_plan(self):
        """Test creating a valid robot plan."""
        plan = RobotPlan(
            task_name="move_red_cup",
            goal="Move red cup to kitchen counter",
            steps=[
                PlanStep(
                    description="Move to table",
                    tool_name="robot_move_to",
                    input_payload={"target_id": "table"},
                ),
            ],
        )

        assert plan.task_name == "move_red_cup"
        assert len(plan.steps) == 1
        assert plan.requires_verification is True

    def test_robot_plan_validation_empty_steps(self):
        """Test that empty steps list raises validation error."""
        with pytest.raises(ValueError, match="Plan must contain at least one step"):
            RobotPlan(
                task_name="test",
                goal="test",
                steps=[],
            )

    def test_robot_plan_default_values(self):
        """Test robot plan default values."""
        plan = RobotPlan(
            steps=[
                PlanStep(
                    description="Test",
                    tool_name="robot_get_state",
                ),
            ],
        )

        assert plan.task_name == "robot_task"
        assert plan.goal == ""
        assert plan.rationale == ""
        assert plan.final_verification == ""
        assert plan.requires_memory is False
        assert plan.baseline == "full_system"


# ----------------------------------------------------------------------
# TestHeuristicPlanner
# ----------------------------------------------------------------------


class TestHeuristicPlanner:
    @pytest.mark.asyncio
    async def test_heuristic_plan_red_cup_task(self, planner):
        """Test heuristic planner generates correct plan for red cup task."""
        plan = await planner.plan(
            task_instruction="Take the red cup on the table and put it on the kitchen counter",
            task_name="move_red_cup",
            current_state={"robot_location": "home"},
        )

        assert plan.task_name == "move_red_cup"
        assert len(plan.steps) > 0

        # Check that plan includes key actions
        tool_names = [step.tool_name for step in plan.steps]
        assert "robot_get_state" in tool_names
        assert "robot_detect_object" in tool_names
        assert "robot_move_to" in tool_names
        assert "robot_grasp_object" in tool_names
        assert "robot_release_object" in tool_names

    @pytest.mark.asyncio
    async def test_heuristic_plan_includes_verification(self, planner):
        """Test that heuristic plan includes verification steps when enabled."""
        plan = await planner.plan(
            task_instruction="Take the red cup",
            current_state={},
        )

        # Should have verification steps
        verification_steps = [
            step for step in plan.steps
            if step.step_type == PlanStepType.VERIFICATION
        ]

        assert len(verification_steps) > 0

    @pytest.mark.asyncio
    async def test_heuristic_plan_without_verification(self, test_settings, mock_memory_client):
        """Test heuristic plan without verification (baseline A)."""
        planner = create_planner(
            settings=test_settings,
            memory_client=mock_memory_client,
            baseline="baseline_a",
        )

        plan = await planner.plan(
            task_instruction="Take the red cup",
            current_state={},
        )

        # Should not have verification steps
        verification_steps = [
            step for step in plan.steps
            if step.step_type == PlanStepType.VERIFICATION
        ]

        assert len(verification_steps) == 0

    @pytest.mark.asyncio
    async def test_heuristic_plan_unknown_task(self, planner):
        """Test heuristic planner with unknown task generates minimal plan."""
        plan = await planner.plan(
            task_instruction="Do something unknown",
            current_state={},
        )

        # Should have at least observation step
        assert len(plan.steps) >= 1
        assert plan.steps[0].tool_name == "robot_get_state"

    @pytest.mark.asyncio
    async def test_heuristic_plan_step_numbering(self, planner):
        """Test that heuristic plan steps are correctly numbered."""
        plan = await planner.plan(
            task_instruction="Take the red cup",
            current_state={},
        )

        for i, step in enumerate(plan.steps, start=1):
            assert step.step_id == i


# ----------------------------------------------------------------------
# TestRecoveryPlanner
# ----------------------------------------------------------------------


class TestRecoveryPlanner:
    @pytest.mark.asyncio
    async def test_recovery_plan_for_grasp_failure(self, planner):
        """Test recovery plan generation for grasp failure."""
        # Create a failed plan
        failed_plan = RobotPlan(
            task_name="move_red_cup",
            goal="Move red cup",
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
                ),
            ],
        )

        failed_step = failed_plan.steps[1]

        recovery_plan = await planner.create_recovery_plan(
            task_instruction="Take the red cup",
            failed_plan=failed_plan,
            failed_step=failed_step,
            error_code="GRASP_FAILED",
            error_message="Grasping failed",
            current_state={"robot_location": "table"},
        )

        assert recovery_plan is not None
        assert len(recovery_plan.steps) > 0

        # Should include observation step for recovery
        tool_names = [step.tool_name for step in recovery_plan.steps]
        assert "robot_get_state" in tool_names

    @pytest.mark.asyncio
    async def test_recovery_plan_for_reach_failure(self, planner):
        """Test recovery plan generation for reach failure."""
        failed_plan = RobotPlan(
            task_name="move_red_cup",
            goal="Move red cup",
            steps=[
                PlanStep(
                    step_id=1,
                    description="Grasp red_cup",
                    tool_name="robot_grasp_object",
                    input_payload={"object_id": "red_cup"},
                ),
            ],
        )

        failed_step = failed_plan.steps[0]

        recovery_plan = await planner.create_recovery_plan(
            task_instruction="Take the red cup",
            failed_plan=failed_plan,
            failed_step=failed_step,
            error_code="OBJECT_NOT_IN_REACH",
            error_message="Object is out of reach",
            current_state={"robot_location": "home"},
        )

        # Should include move_to step to get closer
        tool_names = [step.tool_name for step in recovery_plan.steps]
        assert "robot_move_to" in tool_names

    @pytest.mark.asyncio
    async def test_recovery_plan_preserves_task_info(self, planner):
        """Test that recovery plan preserves task information."""
        failed_plan = RobotPlan(
            task_name="move_red_cup",
            goal="Move red cup to kitchen",
            steps=[
                PlanStep(
                    step_id=1,
                    description="Grasp red_cup",
                    tool_name="robot_grasp_object",
                    input_payload={"object_id": "red_cup"},
                ),
            ],
        )

        failed_step = failed_plan.steps[0]

        recovery_plan = await planner.create_recovery_plan(
            task_instruction="Take the red cup",
            failed_plan=failed_plan,
            failed_step=failed_step,
            error_code="GRASP_FAILED",
            current_state={},
        )

        assert recovery_plan.task_name == "move_red_cup"
        assert "Recovery plan" in recovery_plan.rationale


# ----------------------------------------------------------------------
# TestMemoryRetrieval
# ----------------------------------------------------------------------


class TestMemoryRetrieval:
    @pytest.mark.asyncio
    async def test_planner_retrieves_task_memories(self, planner, mock_memory_client):
        """Test that planner retrieves task memories when enabled."""
        mock_memory_client.retrieve_relevant_experiences.return_value = [
            {
                "id": "exp1",
                "document": "Previous experience with red cup",
                "metadata": {
                    "action": "robot_grasp_object",
                    "outcome": "success",
                    "error_code": "NONE",
                },
            }
        ]

        plan = await planner.plan(
            task_instruction="Take the red cup",
            current_state={},
        )

        # Verify memory was queried
        mock_memory_client.retrieve_relevant_experiences.assert_called_once()
        assert plan is not None

    @pytest.mark.asyncio
    async def test_planner_retrieves_error_memories_for_recovery(self, planner, mock_memory_client):
        """Test that recovery planner retrieves error memories."""
        mock_memory_client.retrieve_error_recovery_experiences.return_value = [
            {
                "id": "exp2",
                "document": "Recovery from grasp failure",
                "metadata": {
                    "action": "robot_grasp_object",
                    "outcome": "recovered",
                    "error_code": "GRASP_FAILED",
                    "resolution": "Retry after repositioning",
                },
            }
        ]

        failed_plan = RobotPlan(
            task_name="test",
            goal="test",
            steps=[
                PlanStep(
                    step_id=1,
                    description="Grasp",
                    tool_name="robot_grasp_object",
                    input_payload={"object_id": "red_cup"},
                ),
            ],
        )

        recovery_plan = await planner.create_recovery_plan(
            task_instruction="Take the red cup",
            failed_plan=failed_plan,
            failed_step=failed_plan.steps[0],
            error_code="GRASP_FAILED",
            current_state={},
        )

        # Verify error memory was queried
        mock_memory_client.retrieve_error_recovery_experiences.assert_called_once()
        assert recovery_plan is not None

    @pytest.mark.asyncio
    async def test_planner_without_memory(self, test_settings):
        """Test planner works without memory client (baseline A/C)."""
        planner = create_planner(
            settings=test_settings,
            memory_client=None,
            baseline="baseline_a",
        )

        plan = await planner.plan(
            task_instruction="Take the red cup",
            current_state={},
        )

        assert plan is not None
        assert len(plan.steps) > 0


# ----------------------------------------------------------------------
# TestHelpers
# ----------------------------------------------------------------------


class TestHelpers:
    def test_slugify_simple(self):
        """Test slugify with simple text."""
        assert slugify("Move red cup") == "move_red_cup"

    def test_slugify_special_characters(self):
        """Test slugify removes special characters."""
        assert slugify("Move the red cup! @#$%") == "move_the_red_cup"

    def test_slugify_multiple_spaces(self):
        """Test slugify handles multiple spaces."""
        assert slugify("Move   the    red     cup") == "move_the_red_cup"

    def test_slugify_empty_string(self):
        """Test slugify with empty string."""
        assert slugify("") == "robot_task"

    def test_slugify_only_special_chars(self):
        """Test slugify with only special characters."""
        assert slugify("!@#$%^&*()") == "robot_task"

    def test_format_memories_empty(self):
        """Test _format_memories with empty list."""
        result = _format_memories([])
        assert result == "No relevant memories."

    def test_format_memories_with_data(self):
        """Test _format_memories with memory data."""
        memories = [
            {
                "id": "exp1",
                "document": "Previous experience",
                "metadata": {
                    "action": "robot_grasp_object",
                    "outcome": "success",
                    "error_code": "NONE",
                    "resolution": "",
                },
            },
            {
                "id": "exp2",
                "document": "Another experience",
                "metadata": {
                    "action": "robot_move_to",
                    "outcome": "failure",
                    "error_code": "MOVE_FAILED",
                    "resolution": "Retry",
                },
            },
        ]

        result = _format_memories(memories)

        assert "1. action=robot_grasp_object" in result
        assert "outcome=success" in result
        assert "2. action=robot_move_to" in result
        assert "outcome=failure" in result
        assert "resolution=Retry" in result

    def test_renumber_steps(self):
        """Test _renumber_steps correctly numbers steps."""
        plan = RobotPlan(
            task_name="test",
            goal="test",
            steps=[
                PlanStep(description="Step 1", tool_name="robot_get_state"),
                PlanStep(description="Step 2", tool_name="robot_move_to"),
                PlanStep(description="Step 3", tool_name="robot_grasp_object"),
            ],
        )

        # Manually set wrong step_ids
        plan.steps[0].step_id = 5
        plan.steps[1].step_id = 10
        plan.steps[2].step_id = 15

        renumbered = _renumber_steps(plan)

        assert renumbered.steps[0].step_id == 1
        assert renumbered.steps[1].step_id == 2
        assert renumbered.steps[2].step_id == 3


# ----------------------------------------------------------------------
# TestToolValidation
# ----------------------------------------------------------------------


class TestToolValidation:
    @pytest.mark.asyncio
    async def test_planner_validates_tools(self, planner):
        """Test that planner validates tool names."""
        # This test ensures the planner doesn't generate plans with invalid tools
        plan = await planner.plan(
            task_instruction="Take the red cup",
            current_state={},
            available_tools=DEFAULT_TOOL_SPECS,
        )

        valid_tools = {tool["name"] for tool in DEFAULT_TOOL_SPECS}

        for step in plan.steps:
            assert step.tool_name in valid_tools
            if step.verification_tool:
                assert step.verification_tool in valid_tools

    @pytest.mark.asyncio
    async def test_planner_with_custom_tools(self, planner):
        """Test planner with custom tool specs."""
        custom_tools = [
            {
                "name": "robot_get_state",
                "description": "Get state",
            },
        ]

        plan = await planner.plan(
            task_instruction="Check state",
            current_state={},
            available_tools=custom_tools,
        )

        # Should only use available tools
        for step in plan.steps:
            assert step.tool_name in {"robot_get_state"}


# ----------------------------------------------------------------------
# TestBaselineBehavior
# ----------------------------------------------------------------------


class TestBaselineBehavior:
    @pytest.mark.asyncio
    async def test_baseline_a_no_verification_no_memory(self, test_settings, mock_memory_client):
        """Baseline A: tools only, no verification, no memory."""
        planner = create_planner(
            settings=test_settings,
            memory_client=mock_memory_client,
            baseline="baseline_a",
        )

        plan = await planner.plan(
            task_instruction="Take the red cup",
            current_state={},
        )

        assert plan.requires_verification is False
        assert plan.requires_memory is False

        # No verification steps
        verification_steps = [
            step for step in plan.steps
            if step.step_type == PlanStepType.VERIFICATION
        ]
        assert len(verification_steps) == 0

        # Memory should not be queried
        mock_memory_client.retrieve_relevant_experiences.assert_not_called()

    @pytest.mark.asyncio
    async def test_baseline_b_memory_no_verification(self, test_settings, mock_memory_client):
        """Baseline B: memory enabled, verification disabled."""
        planner = create_planner(
            settings=test_settings,
            memory_client=mock_memory_client,
            baseline="baseline_b",
        )

        plan = await planner.plan(
            task_instruction="Take the red cup",
            current_state={},
        )

        assert plan.requires_verification is False
        assert plan.requires_memory is True

        # Memory should be queried
        mock_memory_client.retrieve_relevant_experiences.assert_called_once()

    @pytest.mark.asyncio
    async def test_baseline_c_verification_no_memory(self, test_settings, mock_memory_client):
        """Baseline C: verification enabled, memory disabled."""
        planner = create_planner(
            settings=test_settings,
            memory_client=mock_memory_client,
            baseline="baseline_c",
        )

        plan = await planner.plan(
            task_instruction="Take the red cup",
            current_state={},
        )

        assert plan.requires_verification is True
        assert plan.requires_memory is False

        # Should have verification steps
        verification_steps = [
            step for step in plan.steps
            if step.step_type == PlanStepType.VERIFICATION
        ]
        assert len(verification_steps) > 0

        # Memory should not be queried
        mock_memory_client.retrieve_relevant_experiences.assert_not_called()

    @pytest.mark.asyncio
    async def test_full_system_both_enabled(self, test_settings, mock_memory_client):
        """Full system: both memory and verification enabled."""
        planner = create_planner(
            settings=test_settings,
            memory_client=mock_memory_client,
            baseline="full_system",
        )

        plan = await planner.plan(
            task_instruction="Take the red cup",
            current_state={},
        )

        assert plan.requires_verification is True
        assert plan.requires_memory is True

        # Should have verification steps
        verification_steps = [
            step for step in plan.steps
            if step.step_type == PlanStepType.VERIFICATION
        ]
        assert len(verification_steps) > 0

        # Memory should be queried
        mock_memory_client.retrieve_relevant_experiences.assert_called_once()
      
