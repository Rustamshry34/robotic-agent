"""
Tests for MCP robot tools.

This module tests all robot capabilities exposed via MCP:
- robot_get_state
- robot_list_objects
- robot_detect_object
- robot_move_to
- robot_grasp_object
- robot_release_object
- robot_verify_object_location

All tests use SimulatedRobotBackend for deterministic behavior.
"""

from __future__ import annotations

import pytest

from mcp-server.tools import SimulatedRobotBackend


@pytest.fixture
def backend():
    """
    Create a fresh SimulatedRobotBackend for each test.
    """
    return SimulatedRobotBackend()


# ----------------------------------------------------------------------
# robot_get_state
# ----------------------------------------------------------------------


class TestGetState:
    def test_get_state_initial(self, backend: SimulatedRobotBackend):
        """Test initial robot state."""
        result = backend.get_state()

        assert result["success"] is True
        assert result["data"]["robot"]["current_location"] == "home"
        assert result["data"]["robot"]["gripper_open"] is True
        assert result["data"]["robot"]["holding_object"] is None
        assert "red_cup" in result["data"]["objects"]

    def test_get_state_after_move(self, backend: SimulatedRobotBackend):
        """Test state changes after movement."""
        backend.move_to("table")
        result = backend.get_state()

        assert result["success"] is True
        assert result["data"]["robot"]["current_location"] == "table"


# ----------------------------------------------------------------------
# robot_list_objects
# ----------------------------------------------------------------------


class TestListObjects:
    def test_list_objects(self, backend: SimulatedRobotBackend):
        """Test listing all objects."""
        result = backend.list_objects()

        assert result["success"] is True
        assert "red_cup" in result["data"]["object_ids"]
        assert "table" in result["data"]["surfaces"]
        assert "red_cup" in result["data"]["graspable_objects"]
        assert "table" not in result["data"]["graspable_objects"]


# ----------------------------------------------------------------------
# robot_detect_object
# ----------------------------------------------------------------------


class TestDetectObject:
    def test_detect_existing_object(self, backend: SimulatedRobotBackend):
        """Test detecting an existing object."""
        result = backend.detect_object("red_cup")

        assert result["success"] is True
        assert result["data"]["object_id"] == "red_cup"
        assert result["data"]["location"] == "table"
        assert result["data"]["graspable"] is True

    def test_detect_nonexistent_object(self, backend: SimulatedRobotBackend):
        """Test detecting a non-existent object."""
        result = backend.detect_object("blue_bottle")

        assert result["success"] is False
        assert result["error_code"] == "OBJECT_NOT_FOUND"

    def test_detect_surface(self, backend: SimulatedRobotBackend):
        """Test detecting a surface."""
        result = backend.detect_object("table")

        assert result["success"] is True
        assert result["data"]["surface"] is True
        assert result["data"]["graspable"] is False


# ----------------------------------------------------------------------
# robot_move_to
# ----------------------------------------------------------------------


class TestMoveTo:
    def test_move_to_valid_target(self, backend: SimulatedRobotBackend):
        """Test moving to a valid target."""
        result = backend.move_to("table")

        assert result["success"] is True
        assert result["data"]["target_id"] == "table"

        state = backend.get_state()
        assert state["data"]["robot"]["current_location"] == "table"

    def test_move_to_invalid_target(self, backend: SimulatedRobotBackend):
        """Test moving to an invalid target."""
        result = backend.move_to("nonexistent_location")

        assert result["success"] is False
        assert result["error_code"] == "UNKNOWN_TARGET"

    def test_move_with_fault_injection(self, backend: SimulatedRobotBackend):
        """Test movement failure with fault injection."""
        backend.inject_fault("move_to")
        result = backend.move_to("table")

        assert result["success"] is False
        assert result["error_code"] == "MOVE_FAILED"

    def test_move_updates_held_object_location(self, backend: SimulatedRobotBackend):
        """Test that held object location updates after movement."""
        backend.move_to("table")
        backend.grasp_object("red_cup")
        backend.move_to("kitchen_counter")

        state = backend.get_state()
        assert state["data"]["objects"]["red_cup"]["location"] == "gripper"


# ----------------------------------------------------------------------
# robot_grasp_object
# ----------------------------------------------------------------------


class TestGraspObject:
    def test_grasp_valid_object(self, backend: SimulatedRobotBackend):
        """Test grasping a valid object."""
        backend.move_to("table")
        result = backend.grasp_object("red_cup")

        assert result["success"] is True
        assert result["data"]["object_id"] == "red_cup"

        state = backend.get_state()
        assert state["data"]["robot"]["holding_object"] == "red_cup"
        assert state["data"]["robot"]["gripper_open"] is False
        assert state["data"]["objects"]["red_cup"]["location"] == "gripper"

    def test_grasp_nonexistent_object(self, backend: SimulatedRobotBackend):
        """Test grasping a non-existent object."""
        result = backend.grasp_object("blue_bottle")

        assert result["success"] is False
        assert result["error_code"] == "OBJECT_NOT_FOUND"

    def test_grasp_surface(self, backend: SimulatedRobotBackend):
        """Test grasping a surface (should fail)."""
        result = backend.grasp_object("table")

        assert result["success"] is False
        assert result["error_code"] == "CANNOT_GRASP_SURFACE"

    def test_grasp_when_already_holding(self, backend: SimulatedRobotBackend):
        """Test grasping when already holding an object."""
        backend.move_to("table")
        backend.grasp_object("red_cup")
        result = backend.grasp_object("red_cup")

        assert result["success"] is False
        assert result["error_code"] == "ALREADY_GRASPED"

    def test_grasp_out_of_reach(self, backend: SimulatedRobotBackend):
        """Test grasping an object that is out of reach."""
        result = backend.grasp_object("red_cup")

        assert result["success"] is False
        assert result["error_code"] == "OBJECT_NOT_IN_REACH"

    def test_grasp_with_fault_injection(self, backend: SimulatedRobotBackend):
        """Test grasp failure with fault injection."""
        backend.move_to("table")
        backend.inject_fault("grasp_object")
        result = backend.grasp_object("red_cup")

        assert result["success"] is False
        assert result["error_code"] == "GRASP_FAILED"


# ----------------------------------------------------------------------
# robot_release_object
# ----------------------------------------------------------------------


class TestReleaseObject:
    def test_release_valid_target(self, backend: SimulatedRobotBackend):
        """Test releasing an object to a valid target."""
        backend.move_to("table")
        backend.grasp_object("red_cup")
        backend.move_to("kitchen_counter")
        result = backend.release_object("kitchen_counter")

        assert result["success"] is True
        assert result["data"]["released_object"] == "red_cup"
        assert result["data"]["target_id"] == "kitchen_counter"

        state = backend.get_state()
        assert state["data"]["robot"]["holding_object"] is None
        assert state["data"]["robot"]["gripper_open"] is True
        assert state["data"]["objects"]["red_cup"]["location"] == "kitchen_counter"

    def test_release_when_not_holding(self, backend: SimulatedRobotBackend):
        """Test releasing when not holding any object."""
        result = backend.release_object("table")

        assert result["success"] is False
        assert result["error_code"] == "NOT_HOLDING"

    def test_release_to_non_surface(self, backend: SimulatedRobotBackend):
        """Test releasing to a non-surface target."""
        backend.move_to("table")
        backend.grasp_object("red_cup")
        result = backend.release_object("red_cup")

        assert result["success"] is False
        assert result["error_code"] == "TARGET_NOT_SURFACE"

    def test_release_out_of_reach(self, backend: SimulatedRobotBackend):
        """Test releasing to a target that is out of reach."""
        backend.move_to("table")
        backend.grasp_object("red_cup")
        result = backend.release_object("kitchen_counter")

        assert result["success"] is False
        assert result["error_code"] == "TARGET_NOT_IN_REACH"

    def test_release_with_fault_injection(self, backend: SimulatedRobotBackend):
        """Test release failure with fault injection."""
        backend.move_to("table")
        backend.grasp_object("red_cup")
        backend.move_to("kitchen_counter")
        backend.inject_fault("release_object")
        result = backend.release_object("kitchen_counter")

        assert result["success"] is False
        assert result["error_code"] == "RELEASE_FAILED"


# ----------------------------------------------------------------------
# robot_verify_object_location
# ----------------------------------------------------------------------


class TestVerifyObjectLocation:
    def test_verify_correct_location(self, backend: SimulatedRobotBackend):
        """Test verification when object is at expected location."""
        result = backend.verify_object_location("red_cup", "table")

        assert result["success"] is True
        assert result["data"]["verified"] is True
        assert result["data"]["actual_location"] == "table"

    def test_verify_incorrect_location(self, backend: SimulatedRobotBackend):
        """Test verification when object is not at expected location."""
        result = backend.verify_object_location("red_cup", "kitchen_counter")

        assert result["success"] is True
        assert result["data"]["verified"] is False
        assert result["data"]["actual_location"] == "table"
        assert result["data"]["expected_location"] == "kitchen_counter"

    def test_verify_nonexistent_object(self, backend: SimulatedRobotBackend):
        """Test verification of a non-existent object."""
        result = backend.verify_object_location("blue_bottle", "table")

        assert result["success"] is False
        assert result["error_code"] == "OBJECT_NOT_FOUND"

    def test_verify_after_grasp(self, backend: SimulatedRobotBackend):
        """Test verification after grasping."""
        backend.move_to("table")
        backend.grasp_object("red_cup")
        result = backend.verify_object_location("red_cup", "gripper")

        assert result["success"] is True
        assert result["data"]["verified"] is True

    def test_verify_after_release(self, backend: SimulatedRobotBackend):
        """Test verification after releasing."""
        backend.move_to("table")
        backend.grasp_object("red_cup")
        backend.move_to("kitchen_counter")
        backend.release_object("kitchen_counter")
        result = backend.verify_object_location("red_cup", "kitchen_counter")

        assert result["success"] is True
        assert result["data"]["verified"] is True


# ----------------------------------------------------------------------
# Fault injection and recovery
# ----------------------------------------------------------------------


class TestFaultInjection:
    def test_fault_injection_queue(self, backend: SimulatedRobotBackend):
        """Test that fault injection queues a one-time fault."""
        backend.inject_fault("grasp_object")

        # First attempt should fail
        backend.move_to("table")
        result1 = backend.grasp_object("red_cup")
        assert result1["success"] is False
        assert result1["error_code"] == "GRASP_FAILED"

        # Second attempt should succeed (fault consumed)
        result2 = backend.grasp_object("red_cup")
        assert result2["success"] is True

    def test_reset_scene(self, backend: SimulatedRobotBackend):
        """Test scene reset."""
        backend.move_to("table")
        backend.grasp_object("red_cup")
        backend.inject_fault("move_to")

        result = backend.reset_scene()
        assert result["success"] is True

        state = backend.get_state()
        assert state["data"]["robot"]["current_location"] == "home"
        assert state["data"]["robot"]["holding_object"] is None
        assert state["data"]["objects"]["red_cup"]["location"] == "table"


# ----------------------------------------------------------------------
# Complete task scenario
# ----------------------------------------------------------------------


class TestCompleteTask:
    def test_complete_task_move_red_cup(self, backend: SimulatedRobotBackend):
        """Test complete task: move red_cup from table to kitchen_counter."""
        # Initial state
        state = backend.get_state()
        assert state["data"]["objects"]["red_cup"]["location"] == "table"

        # Step 1: Move to table
        result = backend.move_to("table")
        assert result["success"] is True

        # Step 2: Detect red_cup
        result = backend.detect_object("red_cup")
        assert result["success"] is True
        assert result["data"]["location"] == "table"

        # Step 3: Grasp red_cup
        result = backend.grasp_object("red_cup")
        assert result["success"] is True

        # Step 4: Verify grasped
        result = backend.verify_object_location("red_cup", "gripper")
        assert result["success"] is True
        assert result["data"]["verified"] is True

        # Step 5: Move to kitchen_counter
        result = backend.move_to("kitchen_counter")
        assert result["success"] is True

        # Step 6: Release red_cup
        result = backend.release_object("kitchen_counter")
        assert result["success"] is True

        # Step 7: Verify released
        result = backend.verify_object_location("red_cup", "kitchen_counter")
        assert result["success"] is True
        assert result["data"]["verified"] is True

        # Final state
        state = backend.get_state()
        assert state["data"]["objects"]["red_cup"]["location"] == "kitchen_counter"
        assert state["data"]["robot"]["holding_object"] is None
        assert state["data"]["robot"]["gripper_open"] is True

    def test_complete_task_with_recovery(self, backend: SimulatedRobotBackend):
        """Test complete task with fault recovery."""
        # Move to table
        backend.move_to("table")

        # Inject fault for first grasp attempt
        backend.inject_fault("grasp_object")

        # First grasp attempt fails
        result = backend.grasp_object("red_cup")
        assert result["success"] is False
        assert result["error_code"] == "GRASP_FAILED"

        # Recovery: retry grasp
        result = backend.grasp_object("red_cup")
        assert result["success"] is True

        # Continue task
        backend.move_to("kitchen_counter")
        result = backend.release_object("kitchen_counter")
        assert result["success"] is True

        # Verify final state
        result = backend.verify_object_location("red_cup", "kitchen_counter")
        assert result["data"]["verified"] is True
      
