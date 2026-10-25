from __future__ import annotations

import copy
from typing import Any, Protocol, runtime_checkable

from .types import ToolResult


@runtime_checkable
class RobotBackend(Protocol):
    """
    Abstract robot backend interface.

    This allows swapping:
    - simulated backend
    - ROS 2 backend
    - physical robot backend

    without changing MCP tool definitions.
    """

    def get_state(self) -> dict[str, Any]:
        ...

    def list_objects(self) -> dict[str, Any]:
        ...

    def detect_object(self, object_id: str) -> dict[str, Any]:
        ...

    def move_to(self, target_id: str) -> dict[str, Any]:
        ...

    def grasp_object(self, object_id: str) -> dict[str, Any]:
        ...

    def release_object(self, target_id: str) -> dict[str, Any]:
        ...

    def verify_object_location(
        self,
        object_id: str,
        expected_location: str,
        tolerance_meters: float = 0.10,
    ) -> dict[str, Any]:
        ...

    def reset_scene(self) -> dict[str, Any]:
        ...


class SimulatedRobotBackend:
    """
    Deterministic simulated robot environment.

    Scenario:
    - red_cup starts on table
    - robot starts at home
    - target surface is kitchen_counter

    This backend is intentionally deterministic for CI/CD and evaluation.
    """

    def __init__(self) -> None:
        self._initial_world: dict[str, Any] = {
            "robot": {
                "current_location": "home",
                "pose": {"x": 0.0, "y": 0.0, "z": 0.0},
                "gripper_open": True,
                "holding_object": None,
                "last_error": None,
            },
            "objects": {
                "home": {
                    "object_id": "home",
                    "location": "world",
                    "pose": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "surface": True,
                    "graspable": False,
                    "visible": True,
                },
                "table": {
                    "object_id": "table",
                    "location": "world",
                    "pose": {"x": 0.5, "y": 0.0, "z": 0.0},
                    "surface": True,
                    "graspable": False,
                    "visible": True,
                },
                "kitchen_counter": {
                    "object_id": "kitchen_counter",
                    "location": "world",
                    "pose": {"x": 2.0, "y": 0.0, "z": 0.0},
                    "surface": True,
                    "graspable": False,
                    "visible": True,
                },
                "red_cup": {
                    "object_id": "red_cup",
                    "location": "table",
                    "pose": {"x": 0.5, "y": 0.1, "z": 0.1},
                    "surface": False,
                    "graspable": True,
                    "visible": True,
                },
            },
            "reachable": [
                "home",
                "table",
                "kitchen_counter",
                "red_cup",
            ],
        }

        self._faults: dict[str, bool] = {}
        self.world: dict[str, Any] = {}
        self.reset_scene()

    def _ok(
        self,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return ToolResult(
            success=True,
            message=message,
            data=data,
        ).to_dict()

    def _fail(
        self,
        message: str,
        error_code: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.world["robot"]["last_error"] = error_code
        return ToolResult(
            success=False,
            message=message,
            data=data,
            error_code=error_code,
        ).to_dict()

    def _get_object(self, object_id: str) -> dict[str, Any] | None:
        return self.world["objects"].get(object_id)

    def _consume_fault(self, action: str) -> bool:
        """
        Consumes a queued fault for controlled failure experiments.

        Example:
            backend.inject_fault("grasp_object")
            backend.grasp_object("red_cup")  # fails once
        """
        return self._faults.pop(action, False)

    def inject_fault(self, action: str) -> dict[str, Any]:
        """
        Queue a one-time fault for evaluation and recovery experiments.

        Supported actions:
        - move_to
        - grasp_object
        - release_object
        """
        self._faults[action] = True
        return self._ok(
            message="Fault queued.",
            data={"action": action},
        )

    def get_state(self) -> dict[str, Any]:
        robot = copy.deepcopy(self.world["robot"])
        objects = {}

        for object_id, obj in self.world["objects"].items():
            objects[object_id] = {
                "location": obj["location"],
                "pose": copy.deepcopy(obj["pose"]),
                "surface": obj["surface"],
                "graspable": obj["graspable"],
                "visible": obj["visible"],
            }

        return self._ok(
            message="Robot state retrieved.",
            data={
                "robot": robot,
                "objects": objects,
            },
        )

    def list_objects(self) -> dict[str, Any]:
        objects = self.world["objects"]

        return self._ok(
            message="Object list retrieved.",
            data={
                "object_ids": list(objects.keys()),
                "surfaces": [
                    object_id
                    for object_id, obj in objects.items()
                    if obj["surface"]
                ],
                "graspable_objects": [
                    object_id
                    for object_id, obj in objects.items()
                    if obj["graspable"]
                ],
            },
        )

    def detect_object(self, object_id: str) -> dict[str, Any]:
        object_id = object_id.strip()
        obj = self._get_object(object_id)

        if obj is None:
            return self._fail(
                message=f"Object '{object_id}' not found.",
                error_code="OBJECT_NOT_FOUND",
                data={"object_id": object_id},
            )

        if not obj.get("visible", True):
            return self._fail(
                message=f"Object '{object_id}' is not visible.",
                error_code="OBJECT_NOT_VISIBLE",
                data={"object_id": object_id},
            )

        return self._ok(
            message=f"Object '{object_id}' detected.",
            data={
                "object_id": object_id,
                "location": obj["location"],
                "pose": copy.deepcopy(obj["pose"]),
                "surface": obj["surface"],
                "graspable": obj["graspable"],
                "visible": obj["visible"],
            },
        )

    def move_to(self, target_id: str) -> dict[str, Any]:
        target_id = target_id.strip()

        if target_id not in self.world["reachable"]:
            return self._fail(
                message=f"Target '{target_id}' is unknown or unreachable.",
                error_code="UNKNOWN_TARGET",
                data={"target_id": target_id},
            )

        if self._consume_fault("move_to"):
            return self._fail(
                message=f"Movement to '{target_id}' failed due to simulated fault.",
                error_code="MOVE_FAILED",
                data={"target_id": target_id},
            )

        robot = self.world["robot"]
        target_obj = self._get_object(target_id)

        if target_obj is not None:
            robot["pose"] = copy.deepcopy(target_obj["pose"])
        else:
            robot["pose"] = {"x": 0.0, "y": 0.0, "z": 0.0}

        robot["current_location"] = target_id
        robot["last_error"] = None

        if robot["holding_object"] is not None:
            held_object = self._get_object(robot["holding_object"])
            if held_object is not None:
                held_object["location"] = "gripper"
                held_object["pose"] = {
                    "x": robot["pose"]["x"],
                    "y": robot["pose"]["y"],
                    "z": robot["pose"]["z"] + 0.05,
                }

        return self._ok(
            message=f"Robot moved to '{target_id}'.",
            data={
                "target_id": target_id,
                "robot": copy.deepcopy(robot),
            },
        )

    def grasp_object(self, object_id: str) -> dict[str, Any]:
        object_id = object_id.strip()
        obj = self._get_object(object_id)

        if obj is None:
            return self._fail(
                message=f"Object '{object_id}' not found.",
                error_code="OBJECT_NOT_FOUND",
                data={"object_id": object_id},
            )

        if obj["surface"]:
            return self._fail(
                message=f"Object '{object_id}' is a surface and cannot be grasped.",
                error_code="CANNOT_GRASP_SURFACE",
                data={"object_id": object_id},
            )

        if not obj["graspable"]:
            return self._fail(
                message=f"Object '{object_id}' is not graspable.",
                error_code="OBJECT_NOT_GRASPABLE",
                data={"object_id": object_id},
            )

        robot = self.world["robot"]

        if robot["holding_object"] is not None:
            return self._fail(
                message=f"Robot is already holding '{robot['holding_object']}'.",
                error_code="ALREADY_HOLDING",
                data={
                    "holding_object": robot["holding_object"],
                    "requested_object": object_id,
                },
            )

        if not robot["gripper_open"]:
            return self._fail(
                message="Gripper is not open.",
                error_code="GRIPPER_CLOSED",
                data={"object_id": object_id},
            )

        if obj["location"] == "gripper":
            return self._fail(
                message=f"Object '{object_id}' is already grasped.",
                error_code="ALREADY_GRASPED",
                data={"object_id": object_id},
            )

        if robot["current_location"] not in {object_id, obj["location"]}:
            return self._fail(
                message=(
                    f"Object '{object_id}' is at '{obj['location']}' "
                    f"but robot is at '{robot['current_location']}'."
                ),
                error_code="OBJECT_NOT_IN_REACH",
                data={
                    "object_id": object_id,
                    "object_location": obj["location"],
                    "robot_location": robot["current_location"],
                },
            )

        if self._consume_fault("grasp_object"):
            return self._fail(
                message=f"Grasping '{object_id}' failed due to simulated fault.",
                error_code="GRASP_FAILED",
                data={"object_id": object_id},
            )

        obj["location"] = "gripper"
        obj["pose"] = copy.deepcopy(robot["pose"])

        robot["holding_object"] = object_id
        robot["gripper_open"] = False
        robot["last_error"] = None

        return self._ok(
            message=f"Robot grasped '{object_id}'.",
            data={
                "object_id": object_id,
                "robot": copy.deepcopy(robot),
            },
        )

    def release_object(self, target_id: str) -> dict[str, Any]:
        target_id = target_id.strip()
        robot = self.world["robot"]

        if robot["holding_object"] is None:
            return self._fail(
                message="Robot is not holding any object.",
                error_code="NOT_HOLDING",
                data={"target_id": target_id},
            )

        target_obj = self._get_object(target_id)

        if target_obj is None:
            return self._fail(
                message=f"Target '{target_id}' not found.",
                error_code="UNKNOWN_TARGET",
                data={"target_id": target_id},
            )

        if not target_obj["surface"]:
            return self._fail(
                message=f"Target '{target_id}' is not a valid release surface.",
                error_code="TARGET_NOT_SURFACE",
                data={"target_id": target_id},
            )

        if robot["current_location"] not in {
            target_id,
            target_obj["location"],
        }:
            return self._fail(
                message=(
                    f"Robot is at '{robot['current_location']}' "
                    f"but target surface is '{target_id}'."
                ),
                error_code="TARGET_NOT_IN_REACH",
                data={
                    "target_id": target_id,
                    "robot_location": robot["current_location"],
                },
            )

        if self._consume_fault("release_object"):
            return self._fail(
                message=f"Releasing object at '{target_id}' failed due to simulated fault.",
                error_code="RELEASE_FAILED",
                data={"target_id": target_id},
            )

        held_object_id = robot["holding_object"]
        held_object = self._get_object(held_object_id)

        if held_object is None:
            return self._fail(
                message=f"Held object '{held_object_id}' disappeared.",
                error_code="HELD_OBJECT_MISSING",
                data={"held_object_id": held_object_id},
            )

        target_pose = target_obj["pose"]

        held_object["location"] = target_id
        held_object["pose"] = {
            "x": target_pose["x"],
            "y": target_pose["y"],
            "z": target_pose.get("z", 0.0) + 0.05,
        }

        robot["holding_object"] = None
        robot["gripper_open"] = True
        robot["last_error"] = None

        return self._ok(
            message=f"Robot released '{held_object_id}' at '{target_id}'.",
            data={
                "released_object": held_object_id,
                "target_id": target_id,
                "robot": copy.deepcopy(robot),
            },
        )

    def verify_object_location(
        self,
        object_id: str,
        expected_location: str,
        tolerance_meters: float = 0.10,
    ) -> dict[str, Any]:
        object_id = object_id.strip()
        expected_location = expected_location.strip()

        obj = self._get_object(object_id)

        if obj is None:
            return self._fail(
                message=f"Object '{object_id}' not found.",
                error_code="OBJECT_NOT_FOUND",
                data={
                    "object_id": object_id,
                    "expected_location": expected_location,
                },
            )

        actual_location = obj["location"]
        verified = actual_location == expected_location

        data = {
            "object_id": object_id,
            "expected_location": expected_location,
            "actual_location": actual_location,
            "verified": verified,
            "tolerance_meters": tolerance_meters,
        }

        if verified:
            return self._ok(
                message=f"Verification succeeded: '{object_id}' is at '{expected_location}'.",
                data=data,
            )

        return self._ok(
            message=(
                f"Verification failed: '{object_id}' is at "
                f"'{actual_location}', expected '{expected_location}'."
            ),
            data=data,
        )

    def reset_scene(self) -> dict[str, Any]:
        self.world = copy.deepcopy(self._initial_world)
        self._faults = {}

        return self._ok(
            message="Scene reset to initial state.",
            data={"status": "reset"},
        )
