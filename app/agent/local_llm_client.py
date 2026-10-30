"""
Local LLM Client using HuggingFace Transformers.

Runs inference directly on CPU or GPU without external API dependencies.
Model is loaded lazily on first use and cached for subsequent calls.

Key design decisions (MLOps-grade):
- Strict JSON parsing (no heuristic object boundary guessing)
- Explicit prompt budget check (fail loudly instead of silent truncation)
- Qwen3 thinking-channel suppression for structured output
- Greedy decoding + repetition penalty to prevent degenerate loops
- Async timeout wrapper to bound CI/runtime latency
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Type

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


# Default token budget. Prompt + output must fit within this.
# 4096 is a safe ceiling for small models on CPU.
DEFAULT_CONTEXT_BUDGET = 4096

# Hard cap on generation length. Small models rarely need more than
# ~768 tokens for a structured plan; larger values invite repetition.
DEFAULT_MAX_OUTPUT_TOKENS = 768

# Default per-call timeout. CI runners are slow; 120s is generous.
DEFAULT_TIMEOUT_SECONDS = 120


class TransformersLLMClient:
    """
    Lightweight local LLM client using HuggingFace Transformers.

    Features:
    - Lazy model loading (first call loads, subsequent calls reuse)
    - Async wrapper via asyncio.to_thread with explicit timeout
    - Strict JSON schema injection in system prompt
    - Deterministic greedy decoding with repetition penalty
    - Robust JSON parsing with markdown fence unwrapping only
    - Explicit prompt budget enforcement (no silent truncation)
    """

    def __init__(
        self,
        model_name: str = "LiquidAI/LFM2.5-230M",
        device: str = "cpu",
        dtype: str = "float32",
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        context_budget: int = DEFAULT_CONTEXT_BUDGET,
        repetition_penalty: float = 1.1,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.temperature = temperature

        # Hard-cap: Robot planı için 768 token fazlasıyla yeterli.
        # Daha yüksek değerler küçük modellerde tekrar üretimine neden olur.
        if max_tokens > DEFAULT_MAX_OUTPUT_TOKENS:
            logger.warning(
                "max_tokens=%d exceeds safe limit for structured output. "
                "Capping at %d.",
                max_tokens,
                DEFAULT_MAX_OUTPUT_TOKENS,
            )
            max_tokens = DEFAULT_MAX_OUTPUT_TOKENS
            
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.context_budget = context_budget
        self.repetition_penalty = repetition_penalty

        # Lazy-loaded on first call
        self._model = None
        self._tokenizer = None
        self._load_lock = asyncio.Lock()

    def _load_model(self) -> None:
        """
        Load the model and tokenizer. Called once on first use.
        """
        if self._model is not None:
            return

        logger.info(
            "Loading local model: %s on device: %s (dtype: %s)",
            self.model_name,
            self.device,
            self.dtype,
        )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Map dtype string to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.float32)

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # Ensure tokenizer has pad token
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self._model.to(self.device)
        self._model.eval()

        logger.info("Model loaded successfully: %s", self.model_name)

    async def _ensure_loaded(self) -> None:
        """
        Ensure model is loaded, using a lock to prevent concurrent loading.
        """
        if self._model is not None:
            return

        async with self._load_lock:
            # Double-check after acquiring lock
            if self._model is not None:
                return
            # Load in thread to avoid blocking event loop
            await asyncio.to_thread(self._load_model)

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        """
        Apply the model's chat template.

        For Qwen3-family models, the thinking channel is suppressed
        because structured output is incompatible with internal
        reasoning traces.
        """
        try:
            # Qwen3 path: disable thinking channel
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Fallback for models that don't support enable_thinking
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _tokenize_with_budget_check(self, text: str) -> dict[str, Any]:
        """
        Tokenize input and enforce a hard context budget.

        Raises RuntimeError if prompt + max_tokens exceeds budget.
        This is intentional: silent truncation corrupts the task
        (the model then copies state instead of planning).
        """
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=False,  # batch_size=1, padding unnecessary
        )

        input_len = inputs["input_ids"].shape[-1]

        if input_len + self.max_tokens > self.context_budget:
            raise RuntimeError(
                f"Prompt too long: {input_len} input tokens + "
                f"{self.max_tokens} output tokens = "
                f"{input_len + self.max_tokens} > "
                f"budget {self.context_budget}. "
                "Compact the state/tool context before calling the LLM."
            )

        return inputs

    def _generate_sync(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Synchronous generation. Runs in a thread via asyncio.to_thread.

        Decoding strategy:
        - Greedy (do_sample=False) for deterministic structured output
        - repetition_penalty > 1.0 to prevent degenerate loops
        - No truncation (budget enforced upstream)
        """
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self._apply_chat_template(messages)
        inputs = self._tokenize_with_budget_check(text)
        input_length = inputs["input_ids"].shape[1]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Resolve EOS token(s)
        eos_ids = self._tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,                       # greedy decoding
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=eos_ids,
            )

        generated_tokens = outputs[0][input_length:]
        response = self._tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return response.strip()

    @staticmethod
    def _clean_json_output(content: str) -> str:
        """
        Strict JSON extraction.

        Accepted transformations ONLY:
        - strip surrounding whitespace
        - unwrap a single fenced ```json ... ``` block

        Anything else is a model failure and must be reported as such.
        No find("{") / rfind("}") heuristics: they silently accept
        concatenated or truncated JSON fragments, which corrupts
        evaluation results.
        """
        content = content.strip()

        if not content:
            raise RuntimeError("LLM produced empty output.")

        # Unwrap markdown fence if present
        if content.startswith("```"):
            first_newline = content.find("\n")
            if first_newline == -1:
                raise RuntimeError(
                    "Malformed markdown fence in LLM output (no newline)."
                )
            content = content[first_newline + 1:]

            if content.rstrip().endswith("```"):
                content = content.rstrip()[:-3]

            content = content.strip()

        return content

    @staticmethod
    def _parse_strict_json(content: str) -> Any:
        """
        Parse exactly ONE JSON value from the content.

        Trailing characters after the first complete value are treated
        as an error (indicates concatenated outputs or degenerate
        generation), not silently discarded.
        """
        decoder = json.JSONDecoder()
        parsed_data, end_index = decoder.raw_decode(content)

        trailing = content[end_index:].strip()
        if trailing:
            raise RuntimeError(
                f"LLM emitted multiple/concatenated JSON values. "
                f"Trailing content: {trailing[:200]!r}"
            )

        return parsed_data

    async def generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_instruction: str = "",
    ) -> BaseModel:
        """
        Generate a response strictly conforming to the provided Pydantic schema.

        Pipeline:
        1. Build system prompt with JSON schema
        2. Call LLM in a thread with explicit timeout
        3. Strict JSON extraction (no heuristics)
        4. Strict single-value JSON parse
        5. Pydantic validation
        """
        await self._ensure_loaded()

        json_schema = schema.model_json_schema()

        full_system_prompt = f"""{system_instruction}

You MUST respond with a valid JSON object that strictly follows this schema:
{json.dumps(json_schema, indent=2)}

CRITICAL RULES:
1. Output ONLY the raw JSON object.
2. Do NOT include any additional text, explanations, or markdown formatting.
3. Do NOT include ```json or ``` markers.
4. All required fields must be present.
5. Field types must match the schema exactly.
"""

        content: str | None = None

        try:
            # Run synchronous generation in a thread with explicit timeout
            content = await asyncio.wait_for(
                asyncio.to_thread(
                    self._generate_sync,
                    full_system_prompt,
                    prompt,
                ),
                timeout=self.timeout,
            )

            # Strict JSON extraction (no heuristic boundary guessing)
            cleaned = self._clean_json_output(content)

            # Strict single-value parse
            parsed_data = self._parse_strict_json(cleaned)

            # Validate against Pydantic schema
            return schema.model_validate(parsed_data)

        except asyncio.TimeoutError:
            logger.error(
                "Local LLM generation timed out after %s seconds.",
                self.timeout,
            )
            raise RuntimeError(
                f"Local LLM generation timed out after {self.timeout}s."
            )

        except json.JSONDecodeError as e:
            snippet = (content or "")[:500]
            logger.error(
                "Failed to parse JSON from local LLM. Raw output: %s",
                snippet,
            )
            raise RuntimeError(f"Invalid JSON from local LLM: {e}")

        except ValidationError as e:
            logger.error("Pydantic validation failed: %s", e)
            raise RuntimeError(
                f"Local LLM output did not match schema: {e}"
            )

        except RuntimeError:
            # Re-raise our own RuntimeErrors unchanged
            raise

        except Exception as e:
            logger.error("Local LLM generation failed: %s", e)
            raise RuntimeError(f"Local LLM error: {e}")
