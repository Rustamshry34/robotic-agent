"""
Planner for the self-verifying robotic AI agent.

This module converts a natural-language task into a structured robot plan.

The planner supports:
- MCP tool-aware planning
- self-verification steps
- experience memory retrieval
- baseline experiments:
    - baseline_a: tools only
    - baseline_b: tools + memory
    - baseline_c: tools + verification
    - full_system: tools + memory + verification + recovery

Architecture:
    The LLM produces a minimal PlanDraft (just tool_name + args),
    which is then compiled into a full RobotPlan by Python.
    This separation lets small models focus on what they do well:
    choosing the right tool and arguments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.agent.local_llm_client import TransformersLLMClient
from app.core.config import Settings, get_settings
from memory.chroma_client import ChromaMemoryClient, get_chroma_memory


logger = logging.getLogger(__name__)


__all__ = [
    "DEFAULT_TOOL_SPECS",
    "FailureStrategy",
    "PlanDraft",
    "PlannedAction",
    "PlanStep",
    "PlanStepType",
    "PlannerConfig",
    "RobotPlan",
    "RobotPlanner",
    "create_planner",
]


# ----------------------------------------------------------------------
# Planner schemas
# ----------------------------------------------------------------------


class PlanStepType(StrEnum):
    ACTION = "action"
    OBSERVATION = "observation"
    VERIFICATION = "verification"
    RECOVERY = "recovery"


class FailureStrategy(StrEnum):
    RETRY = "retry"
    REPLAN = "replan"
    ABORT = "abort"


class PlanStep(BaseModel):
    """
    A single executable step inside a robot plan.
    """

    step_id: int = Field(default=0, ge=0)
    description: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    step_type: PlanStepType = PlanStepType.ACTION

    input_payload: dict[str, Any] = Field(default_factory=dict)

    expected_state: str = ""

    verification_tool: str | None = None
    verification_payload: dict[str, Any] = Field(default_factory=dict)

    on_failure: FailureStrategy = FailureStrategy.REPLAN
    max_retries: int = Field(default=1, ge=0, le=5)


class RobotPlan(BaseModel):
    """
    Structured plan produced by the planner.

    This plan can be consumed by:
    - RobotExecutor
    - evaluation pipeline
    - PostgreSQL logging layer
    """

    task_name: str = "robot_task"
    goal: str = ""
    rationale: str = ""
    steps: list[PlanStep] = Field(default_factory=list)

    final_verification: str = ""

    requires_memory: bool = False
    requires_verification: bool = True
    baseline: str = "full_system"

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, steps: list[PlanStep]) -> list[PlanStep]:
        if not steps:
            raise ValueError("Plan must contain at least one step.")
        return steps


# ----------------------------------------------------------------------
# NEW: Minimal LLM-friendly draft schema
# ----------------------------------------------------------------------


class PlannedAction(BaseModel):
    """
    Minimal action spec that a small LLM can reliably produce.

    The LLM only decides: which tool + which arguments.
    Everything else (step_id, description, verification, etc.)
    is filled in deterministically by the Python compiler.
    """

    tool_name: str = Field(min_length=1)
    input_payload: dict[str, Any] = Field(default_factory=dict)


class PlanDraft(BaseModel):
    """
    Small-model-friendly output contract.

    Contains ONLY what the LLM must decide:
    - An ordered list of actions (tool + args)

    All structural fields (step_id, description, verification,
    baseline, etc.) are owned by the Python compiler.
    """

    steps: list[PlannedAction] = Field(
        min_length=1,
        max_length=12,
        description="Ordered list of actions to execute.",
    )


class PlannerConfig(BaseModel):
    """
    Planner behavior configuration.

    This is also used to implement baseline experiments.
    """

    enable_memory: bool = True
    enable_verification: bool = True
    baseline: str = "full_system"

    max_plan_steps: int = Field(default=12, ge=1)
    memory_top_k: int = Field(default=5, ge=1)

    @staticmethod
    def from_settings(
        settings: Settings,
        baseline: str | None = None,
    ) -> PlannerConfig:
        """
        Build planner config from app settings.

        Baselines:
            baseline_a: LLM planning + robot tools only
            baseline_b: planning + experience memory
            baseline_c: planning + action verification
            full_system: planning + verification + state-aware memory + recovery
        """
        baseline_name = (baseline or settings.evaluation_baseline).strip().lower()

        if baseline_name in {"a", "baseline_a"}:
            enable_memory = False
            enable_verification = False
            baseline_name = "baseline_a"

        elif baseline_name in {"b", "baseline_b"}:
            enable_memory = True
            enable_verification = False
            baseline_name = "baseline_b"

        elif baseline_name in {"c", "baseline_c"}:
            enable_memory = False
            enable_verification = True
            baseline_name = "baseline_c"

        else:
            enable_memory = True
            enable_verification = True
            baseline_name = "full_system"

        enable_memory = enable_memory and settings.agent_enable_memory
        enable_verification = (
            enable_verification and settings.agent_enable_verification
        )

        return PlannerConfig(
            enable_memory=enable_memory,
            enable_verification=enable_verification,
            baseline=baseline_name,
            max_plan_steps=settings.agent_max_plan_steps,
            memory_top_k=settings.chroma_top_k,
        )


# ----------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------


SYSTEM_PROMPT = """
You are a careful robotic planning agent.

Your job is to convert a natural-language robot task into a safe, executable plan using ONLY the provided MCP robot tools.

CRITICAL RULES:
1. You MUST output ONLY a valid JSON object matching the schema below.
2. NO markdown, NO ```json tags, NO explanations, NO extra text.
3. The 'steps' array MUST NOT be empty. It must contain at least one action.
4. Every action MUST have 'tool_name' (string) and 'input_payload' (object).
5. Use only the available MCP tools listed below.
6. Prefer small, observable, reversible steps.
7. After grasp or release actions, the verification step will be added automatically by the compiler.

AVAILABLE TOOLS:
- robot_get_state: Get current robot state (no args needed, use {}).
- robot_list_objects: List known objects (no args needed, use {}).
- robot_detect_object: Detect a specific object. Args: {"object_id": "<id>"}.
- robot_move_to: Move robot to target. Args: {"target_id": "<id>"}.
- robot_grasp_object: Grasp an object. Args: {"object_id": "<id>"}.
- robot_release_object: Release held object. Args: {"target_id": "<surface_id>"}.
- robot_verify_object_location: Verify object location. Args: {"object_id": "<id>", "expected_location": "<loc>"}.

EXACT OUTPUT FORMAT EXAMPLE:
{
  "steps": [
    {"tool_name": "robot_get_state", "input_payload": {}},
    {"tool_name": "robot_detect_object", "input_payload": {"object_id": "red_cup"}},
    {"tool_name": "robot_move_to", "input_payload": {"target_id": "red_cup"}},
    {"tool_name": "robot_grasp_object", "input_payload": {"object_id": "red_cup"}},
    {"tool_name": "robot_move_to", "input_payload": {"target_id": "kitchen_counter"}},
    {"tool_name": "robot_release_object", "input_payload": {"target_id": "kitchen_counter"}}
  ]
}
""".strip()


HUMAN_PROMPT_TEMPLATE = """
Planning mode: {planning_mode}

Task name: {task_name}

Task instruction: {task_instruction}

Current robot state (compact):
{current_state}

Available MCP tools:
{available_tools}

Relevant memories:
{relevant_memories}

Error context:
{error_context}

Constraints:
- Maximum plan steps: {max_plan_steps}
- Verification enabled: {enable_verification}
- Use only available MCP tools.
- If an error occurred, produce a recovery plan that fixes the failure.

Output ONLY the JSON object. No markdown. No explanations.
""".strip()


# ----------------------------------------------------------------------
# Default MCP tool specs
# ----------------------------------------------------------------------


DEFAULT_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "robot_get_state",
        "description": "Get current robot state, gripper state, held object, and object locations.",
    },
    {
        "name": "robot_list_objects",
        "description": "List known objects, surfaces, and graspable objects.",
    },
    {
        "name": "robot_detect_object",
        "description": "Detect a specific object and return its pose/location.",
        "parameters": {
            "object_id": "string",
        },
    },
    {
        "name": "robot_move_to",
        "description": "Move the robot to a target location or near a target object.",
        "parameters": {
            "target_id": "string",
        },
    },
    {
        "name": "robot_grasp_object",
        "description": "Grasp an object with the robot gripper.",
        "parameters": {
            "object_id": "string",
        },
    },
    {
        "name": "robot_release_object",
        "description": "Release the held object onto a target surface.",
        "parameters": {
            "target_id": "string",
        },
    },
    {
        "name": "robot_verify_object_location",
        "description": "Verify whether an object is at the expected location.",
        "parameters": {
            "object_id": "string",
            "expected_location": "string",
            "tolerance_meters": "float",
        },
    },
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def slugify(text: str) -> str:
    """
    Convert a text into a safe task name.
    """
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "robot_task"


def _to_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def _compact_state(state: dict[str, Any]) -> dict[str, Any]:
    """
    Compact the robot state for LLM consumption.

    Reduces ~2500 tokens of raw state to ~600 tokens by keeping
    only the fields the LLM needs to plan effectively:
    - robot location, gripper state, held object
    - object locations (not full pose/surface info)
    """
    robot = state.get("robot", {}) or {}
    objects = state.get("objects", {}) or {}

    return {
        "robot_location": robot.get("current_location"),
        "holding_object": robot.get("holding_object"),
        "gripper_open": robot.get("gripper_open"),
        "object_locations": {
            name: obj.get("location")
            for name, obj in objects.items()
        },
    }


def _format_memories(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "No relevant memories."

    lines = []

    for index, memory in enumerate(memories, start=1):
        metadata = memory.get("metadata", {})
        document = memory.get("document", "")

        action = metadata.get("action", "unknown")
        outcome = metadata.get("outcome", "unknown")
        error_code = metadata.get("error_code", "NONE")
        resolution = metadata.get("resolution", "")

        line = (
            f"{index}. action={action} "
            f"outcome={outcome} "
            f"error_code={error_code}"
        )

        if resolution:
            line += f" resolution={resolution}"

        if document:
            line += f"\n   memory: {document}"

        lines.append(line)

    return "\n".join(lines)


def _renumber_steps(plan: RobotPlan) -> RobotPlan:
    for index, step in enumerate(plan.steps, start=1):
        step.step_id = index

    return plan


# ----------------------------------------------------------------------
# Planner
# ----------------------------------------------------------------------


class RobotPlanner:
    """
    LLM-backed structured planner for robot tasks.

    Uses a small local LLM via TransformersLLMClient.
    The LLM produces a minimal PlanDraft (just tool_name + args),
    which is then compiled into a full RobotPlan by Python.

    Args:
        strict_llm: If True, LLM failures are raised instead of
            falling back to heuristic planner. Used in evaluation
            jobs to prevent silent fallback from corrupting results.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        memory_client: ChromaMemoryClient | None = None,
        config: PlannerConfig | None = None,
        llm_client: TransformersLLMClient | None = None,
        strict_llm: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.config = config or PlannerConfig.from_settings(self.settings)
        self.strict_llm = strict_llm

        self.memory_client = memory_client

        if self.config.enable_memory and self.memory_client is None:
            try:
                self.memory_client = get_chroma_memory()
            except Exception:
                logger.exception(
                    "Could not initialize ChromaMemoryClient. "
                    "Memory retrieval will be disabled."
                )
                self.memory_client = None

        self.llm_client = llm_client or self._create_llm_client()

    def _create_llm_client(self) -> TransformersLLMClient | None:
        """
        Create the local LLM client.

        Returns None if USE_LLM_IN_TESTS=false (CI ortamında heuristic planner kullanılır).
        """
        import os

        use_llm = os.getenv("USE_LLM_IN_TESTS", "true").lower()
        if use_llm in {"false", "0", "no"}:
            logger.info("USE_LLM_IN_TESTS=false, using heuristic planner")
            return None

        return TransformersLLMClient(
            model_name=self.settings.llm_model,
            device=self.settings.llm_device,
            dtype=self.settings.llm_dtype,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def plan(
        self,
        task_instruction: str,
        task_name: str | None = None,
        current_state: dict[str, Any] | None = None,
        available_tools: list[dict[str, Any]] | None = None,
    ) -> RobotPlan:
        """
        Create a structured robot plan for a task.
        """
        task_name = task_name or slugify(task_instruction)
        current_state = current_state or {}
        available_tools = available_tools or DEFAULT_TOOL_SPECS

        memories: list[dict[str, Any]] = []

        if self.config.enable_memory:
            memories = await self._retrieve_task_memories(
                task_instruction=task_instruction,
                current_state=current_state,
            )

        if self.llm_client is None:
            plan = self._heuristic_plan(
                task_instruction=task_instruction,
                task_name=task_name,
                current_state=current_state,
            )
        else:
            try:
                plan = await self._llm_plan(
                    planning_mode="initial_plan",
                    task_instruction=task_instruction,
                    task_name=task_name,
                    current_state=current_state,
                    available_tools=available_tools,
                    memories=memories,
                )
            except Exception:
                logger.exception("LLM planning failed.")

                if self.strict_llm:
                    raise

                logger.warning("Falling back to heuristic planner.")
                plan = self._heuristic_plan(
                    task_instruction=task_instruction,
                    task_name=task_name,
                    current_state=current_state,
                )

        plan = self._apply_baseline_policy(plan)
        plan = _renumber_steps(plan)

        self._validate_plan(
            plan=plan,
            available_tools=available_tools,
        )

        return plan

    async def create_recovery_plan(
        self,
        task_instruction: str,
        failed_plan: RobotPlan,
        failed_step: PlanStep,
        error_code: str,
        error_message: str | None = None,
        current_state: dict[str, Any] | None = None,
        available_tools: list[dict[str, Any]] | None = None,
        task_name: str | None = None,
    ) -> RobotPlan:
        """
        Create a recovery plan after a failed step.
        """
        task_name = task_name or failed_plan.task_name
        current_state = current_state or {}
        available_tools = available_tools or DEFAULT_TOOL_SPECS

        memories: list[dict[str, Any]] = []

        if self.config.enable_memory:
            memories = await self._retrieve_error_memories(
                action=failed_step.tool_name,
                error_code=error_code,
                current_state=current_state,
            )

        if self.llm_client is None:
            plan = self._heuristic_recovery_plan(
                failed_plan=failed_plan,
                failed_step=failed_step,
                error_code=error_code,
                error_message=error_message,
            )
        else:
            try:
                plan = await self._llm_recovery_plan(
                    task_instruction=task_instruction,
                    task_name=task_name,
                    current_state=current_state,
                    available_tools=available_tools,
                    memories=memories,
                    failed_plan=failed_plan,
                    failed_step=failed_step,
                    error_code=error_code,
                    error_message=error_message,
                )
            except Exception:
                logger.exception("LLM recovery planning failed.")

                if self.strict_llm:
                    raise

                logger.warning("Falling back to heuristic recovery planner.")
                plan = self._heuristic_recovery_plan(
                    failed_plan=failed_plan,
                    failed_step=failed_step,
                    error_code=error_code,
                    error_message=error_message,
                )

        plan = self._apply_baseline_policy(plan)
        plan = _renumber_steps(plan)

        self._validate_plan(
            plan=plan,
            available_tools=available_tools,
        )

        return plan

    # ------------------------------------------------------------------
    # LLM planning (via PlanDraft)
    # ------------------------------------------------------------------

    async def _llm_plan(
        self,
        planning_mode: str,
        task_instruction: str,
        task_name: str,
        current_state: dict[str, Any],
        available_tools: list[dict[str, Any]],
        memories: list[dict[str, Any]],
    ) -> RobotPlan:
        """
        Generate a plan using the local LLM via PlanDraft.
        """
        if self.llm_client is None:
            raise RuntimeError("LLM client is not available.")

        prompt = HUMAN_PROMPT_TEMPLATE.format(
            planning_mode=planning_mode,
            task_name=task_name,
            task_instruction=task_instruction,
            current_state=_to_json(_compact_state(current_state)),
            available_tools=_to_json(available_tools),
            relevant_memories=_format_memories(memories),
            error_context="None",
            max_plan_steps=self.config.max_plan_steps,
            enable_verification=self.config.enable_verification,
        )

        # LLM produces a minimal PlanDraft
        draft = await self.llm_client.generate_structured(
            prompt=prompt,
            schema=PlanDraft,
            system_instruction=SYSTEM_PROMPT,
        )

        # Python compiles PlanDraft into full RobotPlan
        plan = self._compile_draft(
            draft=draft,
            task_instruction=task_instruction,
            task_name=task_name,
        )

        return plan

    async def _llm_recovery_plan(
        self,
        task_instruction: str,
        task_name: str,
        current_state: dict[str, Any],
        available_tools: list[dict[str, Any]],
        memories: list[dict[str, Any]],
        failed_plan: RobotPlan,
        failed_step: PlanStep,
        error_code: str,
        error_message: str | None,
    ) -> RobotPlan:
        """
        Generate a recovery plan using the local LLM via PlanDraft.
        """
        if self.llm_client is None:
            raise RuntimeError("LLM client is not available.")

        error_context = {
            "failed_step_id": failed_step.step_id,
            "failed_tool": failed_step.tool_name,
            "failed_step_description": failed_step.description,
            "failed_step_input": failed_step.input_payload,
            "error_code": error_code,
            "error_message": error_message,
            "previous_plan": [
                {"tool_name": s.tool_name, "input_payload": s.input_payload}
                for s in failed_plan.steps
            ],
        }

        prompt = HUMAN_PROMPT_TEMPLATE.format(
            planning_mode="recovery_plan",
            task_name=task_name,
            task_instruction=task_instruction,
            current_state=_to_json(_compact_state(current_state)),
            available_tools=_to_json(available_tools),
            relevant_memories=_format_memories(memories),
            error_context=_to_json(error_context),
            max_plan_steps=self.config.max_plan_steps,
            enable_verification=self.config.enable_verification,
        )

        draft = await self.llm_client.generate_structured(
            prompt=prompt,
            schema=PlanDraft,
            system_instruction=SYSTEM_PROMPT,
        )

        plan = self._compile_draft(
            draft=draft,
            task_instruction=task_instruction,
            task_name=task_name,
            is_recovery=True,
        )
        plan.rationale = (
            f"Recovery plan generated after error_code={error_code}. "
            f"error_message={error_message or 'unknown'}"
        )

        return plan

    # ------------------------------------------------------------------
    # Draft compiler: PlanDraft -> RobotPlan
    # ------------------------------------------------------------------

    def _compile_draft(
        self,
        draft: PlanDraft,
        task_instruction: str,
        task_name: str,
        is_recovery: bool = False,
    ) -> RobotPlan:
        """
        Turn a minimal LLM draft into a validated RobotPlan.

        Deterministic fields are owned by code, NOT by the LLM:
        - step_id (sequential numbering)
        - description (auto-generated from tool + args)
        - step_type (inferred from tool name)
        - verification_tool / verification_payload (added automatically
          after grasp/release when verification is enabled)
        - baseline, requires_memory, requires_verification (from config)

        This separation lets a small model focus on what it does well:
        choosing the right tool and arguments.
        """
        steps: list[PlanStep] = []

        for action in draft.steps:
            # Infer step type from tool name
            if action.tool_name in {
                "robot_get_state",
                "robot_list_objects",
                "robot_detect_object",
            }:
                step_type = PlanStepType.OBSERVATION
            elif is_recovery:
                step_type = PlanStepType.RECOVERY
            else:
                step_type = PlanStepType.ACTION

            # Auto-generate description
            args_str = ", ".join(
                f"{k}={v!r}" for k, v in action.input_payload.items()
            )
            description = (
                f"{action.tool_name}({args_str})" if args_str else action.tool_name
            )

            # Determine if this action needs auto-verification
            needs_verify = (
                self.config.enable_verification
                and action.tool_name in {
                    "robot_grasp_object",
                    "robot_release_object",
                }
            )

            verification_tool = None
            verification_payload: dict[str, Any] = {}

            if needs_verify:
                verification_tool = "robot_verify_object_location"
                if action.tool_name == "robot_grasp_object":
                    object_id = action.input_payload.get("object_id", "object")
                    verification_payload = {
                        "object_id": object_id,
                        "expected_location": "gripper",
                    }
                elif action.tool_name == "robot_release_object":
                    target_id = action.input_payload.get("target_id", "surface")
                    # We don't know the exact object being released at plan time,
                    # but we can use a placeholder. The executor will fill it in.
                    verification_payload = {
                        "object_id": "__held_object__",
                        "expected_location": target_id,
                    }

            steps.append(
                PlanStep(
                    step_id=len(steps) + 1,
                    description=description,
                    tool_name=action.tool_name,
                    step_type=step_type,
                    input_payload=action.input_payload,
                    expected_state=f"{action.tool_name} executed.",
                    verification_tool=verification_tool,
                    verification_payload=verification_payload,
                )
            )

            # Auto-append verification step if needed
            if needs_verify:
                steps.append(
                    PlanStep(
                        step_id=len(steps) + 1,
                        description=f"Verify result of {action.tool_name}.",
                        tool_name="robot_verify_object_location",
                        step_type=PlanStepType.VERIFICATION,
                        input_payload=verification_payload,
                        expected_state="Verification passed.",
                    )
                )

        return RobotPlan(
            task_name=task_name,
            goal=task_instruction,
            rationale="LLM-selected action sequence, compiled by planner.",
            steps=steps,
            baseline=self.config.baseline,
            requires_memory=self.config.enable_memory,
            requires_verification=self.config.enable_verification,
        )

    # ------------------------------------------------------------------
    # Memory retrieval
    # ------------------------------------------------------------------

    async def _retrieve_task_memories(
        self,
        task_instruction: str,
        current_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not self.memory_client:
            return []

        query_text = "\n".join(
            [
                f"Task: {task_instruction}",
                f"State: {_to_json(_compact_state(current_state))}",
            ]
        )

        try:
            return await asyncio.to_thread(
                self.memory_client.retrieve_relevant_experiences,
                query_text,
                self.config.memory_top_k,
            )
        except Exception:
            logger.exception("Memory retrieval failed.")
            return []

    async def _retrieve_error_memories(
        self,
        action: str,
        error_code: str,
        current_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not self.memory_client:
            return []

        try:
            return await asyncio.to_thread(
                self.memory_client.retrieve_error_recovery_experiences,
                action,
                error_code,
                current_state,
                self.config.memory_top_k,
            )
        except Exception:
            logger.exception("Error memory retrieval failed.")
            return []

    # ------------------------------------------------------------------
    # Baseline policy and validation
    # ------------------------------------------------------------------

    def _apply_baseline_policy(self, plan: RobotPlan) -> RobotPlan:
        """
        Enforce baseline behavior on the generated plan.
        """
        plan.baseline = self.config.baseline
        plan.requires_memory = self.config.enable_memory
        plan.requires_verification = self.config.enable_verification

        if not self.config.enable_verification:
            plan.steps = [
                step
                for step in plan.steps
                if step.step_type != PlanStepType.VERIFICATION
            ]

            for step in plan.steps:
                step.verification_tool = None
                step.verification_payload = {}

            plan.final_verification = ""

        return plan

    def _validate_plan(
        self,
        plan: RobotPlan,
        available_tools: list[dict[str, Any]],
    ) -> None:
        if not plan.steps:
            raise ValueError("Plan has no steps.")

        if len(plan.steps) > self.config.max_plan_steps:
            raise ValueError(
                f"Plan has {len(plan.steps)} steps, "
                f"but maximum allowed is {self.config.max_plan_steps}."
            )

        tool_names = {
            tool.get("name")
            for tool in available_tools
            if isinstance(tool, dict) and tool.get("name")
        }

        if not tool_names:
            return

        for step in plan.steps:
            if step.tool_name not in tool_names:
                raise ValueError(
                    f"Plan step uses unknown tool '{step.tool_name}'."
                )

            if step.verification_tool and step.verification_tool not in tool_names:
                raise ValueError(
                    f"Plan step uses unknown verification tool "
                    f"'{step.verification_tool}'."
                )

    # ------------------------------------------------------------------
    # Heuristic planners
    # ------------------------------------------------------------------

    def _heuristic_plan(
        self,
        task_instruction: str,
        task_name: str,
        current_state: dict[str, Any],
    ) -> RobotPlan:
        """
        Deterministic fallback planner.

        This is mainly useful for:
        - local development
        - CI tests
        - LLM-free smoke tests
        """
        instruction = task_instruction.lower()

        steps: list[PlanStep] = []

        steps.append(
            PlanStep(
                description="Get current robot state.",
                tool_name="robot_get_state",
                step_type=PlanStepType.OBSERVATION,
                expected_state="Robot state is known.",
            )
        )

        if "red cup" in instruction or "kirmizi bardak" in instruction:
            object_id = "red_cup"
            target_id = "kitchen_counter"

            steps.extend(
                [
                    PlanStep(
                        description=f"Detect {object_id}.",
                        tool_name="robot_detect_object",
                        step_type=PlanStepType.OBSERVATION,
                        input_payload={"object_id": object_id},
                        expected_state=f"{object_id} is visible.",
                    ),
                    PlanStep(
                        description=f"Move near {object_id}.",
                        tool_name="robot_move_to",
                        step_type=PlanStepType.ACTION,
                        input_payload={"target_id": object_id},
                        expected_state=f"Robot is near {object_id}.",
                    ),
                    PlanStep(
                        description=f"Grasp {object_id}.",
                        tool_name="robot_grasp_object",
                        step_type=PlanStepType.ACTION,
                        input_payload={"object_id": object_id},
                        expected_state=f"{object_id} is held by the robot.",
                        verification_tool=(
                            "robot_verify_object_location"
                            if self.config.enable_verification
                            else None
                        ),
                        verification_payload=(
                            {
                                "object_id": object_id,
                                "expected_location": "gripper",
                            }
                            if self.config.enable_verification
                            else {}
                        ),
                    ),
                ]
            )

            if self.config.enable_verification:
                steps.append(
                    PlanStep(
                        description=f"Verify {object_id} is in gripper.",
                        tool_name="robot_verify_object_location",
                        step_type=PlanStepType.VERIFICATION,
                        input_payload={
                            "object_id": object_id,
                            "expected_location": "gripper",
                        },
                        expected_state=f"{object_id} is in gripper.",
                    )
                )

            steps.extend(
                [
                    PlanStep(
                        description=f"Move to {target_id}.",
                        tool_name="robot_move_to",
                        step_type=PlanStepType.ACTION,
                        input_payload={"target_id": target_id},
                        expected_state=f"Robot is at {target_id}.",
                    ),
                    PlanStep(
                        description=f"Release {object_id} at {target_id}.",
                        tool_name="robot_release_object",
                        step_type=PlanStepType.ACTION,
                        input_payload={"target_id": target_id},
                        expected_state=f"{object_id} is at {target_id}.",
                        verification_tool=(
                            "robot_verify_object_location"
                            if self.config.enable_verification
                            else None
                        ),
                        verification_payload=(
                            {
                                "object_id": object_id,
                                "expected_location": target_id,
                            }
                            if self.config.enable_verification
                            else {}
                        ),
                    ),
                ]
            )

            if self.config.enable_verification:
                steps.append(
                    PlanStep(
                        description=(
                            f"Verify {object_id} is at {target_id}."
                        ),
                        tool_name="robot_verify_object_location",
                        step_type=PlanStepType.VERIFICATION,
                        input_payload={
                            "object_id": object_id,
                            "expected_location": target_id,
                        },
                        expected_state=f"{object_id} is at {target_id}.",
                    )
                )

            final_verification = (
                f"{object_id} is at {target_id}."
                if self.config.enable_verification
                else ""
            )

        else:
            steps.append(
                PlanStep(
                    description="List known objects.",
                    tool_name="robot_list_objects",
                    step_type=PlanStepType.OBSERVATION,
                    expected_state="Known objects are listed.",
                )
            )

            final_verification = ""

        return RobotPlan(
            task_name=task_name,
            goal=task_instruction,
            rationale="Deterministic heuristic plan.",
            steps=steps,
            final_verification=final_verification,
            requires_memory=self.config.enable_memory,
            requires_verification=self.config.enable_verification,
            baseline=self.config.baseline,
        )

    def _heuristic_recovery_plan(
        self,
        failed_plan: RobotPlan,
        failed_step: PlanStep,
        error_code: str,
        error_message: str | None,
    ) -> RobotPlan:
        """
        Deterministic fallback recovery planner.
        """
        plan = failed_plan.model_copy(deep=True)
        steps = list(plan.steps)

        failed_index = next(
            (
                index
                for index, step in enumerate(steps)
                if step.step_id == failed_step.step_id
            ),
            None,
        )

        if failed_index is None:
            failed_index = len(steps)

        correction_steps: list[PlanStep] = []

        if error_code in {
            "OBJECT_NOT_IN_REACH",
            "TARGET_NOT_IN_REACH",
            "UNKNOWN_TARGET",
        }:
            target_id = (
                failed_step.input_payload.get("object_id")
                or failed_step.input_payload.get("target_id")
                or "home"
            )

            correction_steps.append(
                PlanStep(
                    description=f"Move to {target_id} before retrying.",
                    tool_name="robot_move_to",
                    step_type=PlanStepType.RECOVERY,
                    input_payload={"target_id": target_id},
                    expected_state=f"Robot is near {target_id}.",
                    on_failure=FailureStrategy.REPLAN,
                )
            )

        elif error_code in {
            "GRASP_FAILED",
            "RELEASE_FAILED",
        }:
            correction_steps.append(
                PlanStep(
                    description="Observe robot state before retrying.",
                    tool_name="robot_get_state",
                    step_type=PlanStepType.OBSERVATION,
                    expected_state="Robot state is known.",
                )
            )

        else:
            correction_steps.append(
                PlanStep(
                    description="Observe robot state after error.",
                    tool_name="robot_get_state",
                    step_type=PlanStepType.OBSERVATION,
                    expected_state="Robot state is known.",
                )
            )

        failed_step.max_retries = min(failed_step.max_retries + 1, 5)
        failed_step.on_failure = FailureStrategy.REPLAN

        steps[failed_index:failed_index] = correction_steps

        plan.steps = steps
        plan.rationale = (
            f"Recovery plan generated after error_code={error_code}. "
            f"error_message={error_message or 'unknown'}"
        )

        return plan


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def create_planner(
    settings: Settings | None = None,
    memory_client: ChromaMemoryClient | None = None,
    baseline: str | None = None,
    llm_client: TransformersLLMClient | None = None,
    strict_llm: bool = False,
) -> RobotPlanner:
    """
    Create a RobotPlanner.

    Args:
        settings:
            Optional app settings. If not provided, uses get_settings().

        memory_client:
            Optional Chroma memory client. Useful for tests.

        baseline:
            Optional baseline override:
            - baseline_a
            - baseline_b
            - baseline_c
            - full_system

        llm_client:
            Optional LLM client. Useful for tests.

        strict_llm:
            If True, LLM failures raise instead of falling back.
            Used in evaluation jobs to prevent silent fallback
            from corrupting results.
    """
    settings = settings or get_settings()

    config = PlannerConfig.from_settings(
        settings=settings,
        baseline=baseline,
    )

    return RobotPlanner(
        settings=settings,
        memory_client=memory_client,
        config=config,
        llm_client=llm_client,
        strict_llm=strict_llm,
    )
