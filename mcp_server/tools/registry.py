from __future__ import annotations

from typing import Any, Optional

from .backend import RobotBackend, SimulatedRobotBackend


def register_tools(
    mcp: Any,
    backend: Optional[RobotBackend] = None,
) -> RobotBackend:
    """
    Registers robot tools with an MCP server instance.

    Usage:
        from mcp.server.fastmcp import FastMCP
        from mcp_server.tools import register_tools

        mcp = FastMCP("robotic-ai-agent")
        backend = register_tools(mcp)
    """

    backend = backend or SimulatedRobotBackend()

    @mcp.tool()
    def robot_get_state() -> dict[str, Any]:
        """
        Get the current robot state.

        Returns robot location, gripper state, held object, last error,
        and all known object locations.
        """
        return backend.get_state()

    @mcp.tool()
    def robot_list_objects() -> dict[str, Any]:
        """
        List all known objects in the environment.

        Returns object IDs, surfaces, and graspable objects.
        """
        return backend.list_objects()

    @mcp.tool()
    def robot_detect_object(object_id: str) -> dict[str, Any]:
        """
        Detect a specific object in the environment.

        Args:
            object_id: Object identifier, e.g. 'red_cup'.

        Returns:
            Object visibility, pose, location, and graspability.
        """
        return backend.detect_object(object_id)

    @mcp.tool()
    def robot_move_to(target_id: str) -> dict[str, Any]:
        """
        Move the robot to a target location or near a target object.

        Args:
            target_id: Target identifier, e.g. 'table', 'kitchen_counter', 'red_cup'.
        """
        return backend.move_to(target_id)

    @mcp.tool()
    def robot_grasp_object(object_id: str) -> dict[str, Any]:
        """
        Grasp an object with the robot gripper.

        Args:
            object_id: Object identifier, e.g. 'red_cup'.
        """
        return backend.grasp_object(object_id)

    @mcp.tool()
    def robot_release_object(target_id: str) -> dict[str, Any]:
        """
        Release the currently held object onto a target surface.

        Args:
            target_id: Target surface identifier, e.g. 'kitchen_counter'.
        """
        return backend.release_object(target_id)

    @mcp.tool()
    def robot_verify_object_location(
        object_id: str,
        expected_location: str,
        tolerance_meters: float = 0.10,
    ) -> dict[str, Any]:
        """
        Verify whether an object is at the expected location.

        This tool is used for self-verification after action execution.

        Args:
            object_id: Object identifier, e.g. 'red_cup'.
            expected_location: Expected location, e.g. 'kitchen_counter' or 'gripper'.
            tolerance_meters: Pose tolerance reserved for future pose-based verification.
        """
        return backend.verify_object_location(
            object_id=object_id,
            expected_location=expected_location,
            tolerance_meters=tolerance_meters,
        )

    return backend
