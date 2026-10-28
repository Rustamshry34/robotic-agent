"""
Local LLM Client using HuggingFace Transformers.

Runs inference directly on CPU or GPU without external API dependencies.
Model is loaded lazily on first use and cached for subsequent calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Type

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class TransformersLLMClient:
    """
    Lightweight local LLM client using HuggingFace Transformers.
    
    Features:
    - Lazy model loading (first call loads, subsequent calls reuse)
    - Async wrapper via asyncio.to_thread
    - JSON schema injection in system prompt
    - Robust JSON parsing with markdown cleanup
    """

    def __init__(
        self,
        model_name: str = "LiquidAI/LFM2.5-230M",
        device: str = "cpu",
        dtype: str = "float32",
        temperature: float = 0.0,
        max_tokens: int = 8048,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.temperature = temperature
        self.max_tokens = max_tokens

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

    def _generate_sync(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Synchronous generation. Runs in a thread via asyncio.to_thread.
        """
        import torch

        # Build chat messages (ChatML format for broad compatibility)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Apply chat template if available, otherwise fallback to simple format
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                # Fallback if chat template fails
                text = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
        else:
            text = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=max(self.temperature, 0.01),  # Avoid 0.0 issues
                do_sample=self.temperature > 0.01,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self._tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return response.strip()

    async def generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_instruction: str = "",
    ) -> BaseModel:
        """
        Generate a response strictly conforming to the provided Pydantic schema.
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

        try:
            # Run synchronous generation in a thread
            content = await asyncio.wait_for(
                asyncio.to_thread(
                    self._generate_sync,
                    full_system_prompt,
                    prompt,
                ),
            )

            # Clean markdown code blocks if present
            content = self._clean_json_output(content)

            # Parse JSON
            parsed_data = json.loads(content)

            # Validate against Pydantic schema
            return schema.model_validate(parsed_data)

        except asyncio.TimeoutError:
            logger.error("Local LLM generation timed out after %s seconds", self.timeout)
            raise RuntimeError(f"Local LLM timeout after {self.timeout}s")
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from local LLM. Raw output: %s", content[:500])
            raise RuntimeError(f"Invalid JSON from local LLM: {e}")
        except ValidationError as e:
            logger.error("Pydantic validation failed: %s", e)
            raise RuntimeError(f"Local LLM output did not match schema: {e}")
        except Exception as e:
            logger.error("Local LLM generation failed: %s", e)
            raise RuntimeError(f"Local LLM error: {e}")

    @staticmethod
    def _clean_json_output(content: str) -> str:
        """
        Remove markdown code blocks and other artifacts from LLM output.
        """
        content = content.strip()

        # Remove ```json ... ```
        if content.startswith("```json"):
            content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
        elif content.startswith("```"):
            content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

        # Find first { and last } to extract JSON object
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start:end + 1]

        return content.strip()
