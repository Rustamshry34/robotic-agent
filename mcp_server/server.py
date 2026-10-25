#!/usr/bin/env python3
"""
MCP server for the self-verifying robotic AI agent.

This server exposes robot capabilities as MCP tools.

Run from repository root:

    python -m mcp_server.server

Or with options:

    python -m mcp_server.server --backend simulated --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Allow running both as a module and as a script.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from mcp_server.tools import (
        RobotBackend,
        SimulatedRobotBackend,
        register_tools,
    )
else:
    from .tools import (
        RobotBackend,
        SimulatedRobotBackend,
        register_tools,
    )

from mcp.server.fastmcp import FastMCP


SERVER_NAME = "robotic-ai-agent-mcp"

logger = logging.getLogger("mcp_server.server")


def configure_logging(level: str) -> None:
    """
    Configure logging for MCP stdio transport.

    Important:
        MCP stdio transport uses stdout for protocol messages.
        Therefore logs must be written to stderr.
    """
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stderr,
    )


def create_backend(backend_name: str) -> RobotBackend:
    """
    Factory for robot backends.

    Currently supported:
        - simulated

    Future:
        - ros2
        - physical_robot
    """
    backend_name = backend_name.strip().lower()

    if backend_name == "simulated":
        logger.info("Using SimulatedRobotBackend.")
        return SimulatedRobotBackend()

    raise ValueError(
        f"Unknown backend: '{backend_name}'. "
        "Available backends: 'simulated'."
    )


def build_server(backend_name: str) -> tuple[FastMCP, RobotBackend]:
    """
    Build MCP server and register robot tools.
    """
    backend = create_backend(backend_name)

    server = FastMCP(SERVER_NAME)

    register_tools(
        mcp=server,
        backend=backend,
    )

    logger.info("MCP server built successfully.")
    logger.info("Registered robot tools.")

    return server, backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Robotic AI Agent MCP server.",
    )

    parser.add_argument(
        "--backend",
        default=os.getenv("MCP_ROBOT_BACKEND", "simulated"),
        choices=["simulated"],
        help=(
            "Robot backend to use. "
            "Can also be set with MCP_ROBOT_BACKEND environment variable."
        ),
    )

    parser.add_argument(
        "--transport",
        default=os.getenv("MCP_TRANSPORT", "stdio"),
        choices=["stdio"],
        help=(
            "MCP transport. "
            "Currently stdio is the default and recommended mode. "
            "Can also be set with MCP_TRANSPORT environment variable."
        ),
    )

    parser.add_argument(
        "--log-level",
        default=os.getenv("MCP_LOG_LEVEL", "INFO"),
        help=(
            "Logging level. Examples: DEBUG, INFO, WARNING, ERROR. "
            "Can also be set with MCP_LOG_LEVEL environment variable."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    configure_logging(args.log_level)

    logger.info("Starting MCP server: %s", SERVER_NAME)
    logger.info("Backend: %s", args.backend)
    logger.info("Transport: %s", args.transport)
    logger.info("Log level: %s", args.log_level)

    server, backend = build_server(args.backend)

    # The backend is returned here for future extensions such as
    # health checks, admin endpoints, or evaluation hooks.
    _ = backend

    try:
        if args.transport == "stdio":
            # FastMCP default transport is stdio.
            # This is the safest mode for local MCP clients and CI.
            server.run()
        else:
            raise SystemExit(
                f"Transport '{args.transport}' is not configured yet. "
                "Use 'stdio' for now."
            )

    except KeyboardInterrupt:
        logger.info("MCP server stopped by user.")


if __name__ == "__main__":
    main()
