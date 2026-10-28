"""
Local LLM Client using vLLM (OpenAI Compatible API).
"""

import json
import logging
from typing import Type

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class LocalLLMClient:
    """
    Lightweight async client for local vLLM instances.
    Leverages vLLM's native JSON output mode for structured planning.
    """

    def __init__(
        self,
        api_key: str = "EMPTY",
        model: str = "LiquidAI/LFM2.5-230M",
        base_url: str = "http://localhost:8000/v1",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: int = 60,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # vLLM, OpenAI API'sini taklit eder
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    async def generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_instruction: str = "",
    ) -> BaseModel:
        """
        Generate a response strictly conforming to the provided Pydantic schema.
        """
        json_schema = schema.model_json_schema()
        
        full_system_prompt = f"""{system_instruction}

You MUST respond with a valid JSON object that strictly follows this schema:
{json.dumps(json_schema, indent=2)}

CRITICAL RULES:
1. Output ONLY the raw JSON object.
2. Do NOT include any additional text, explanations, or markdown formatting.
3. All required fields must be present.
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                # vLLM native JSON mode (OpenAI compatible)
                response_format={"type": "json_object"}, 
            )

            content = response.choices[0].message.content.strip()
            
            # 1. Parse JSON
            parsed_data = json.loads(content)
            
            # 2. Validate against Pydantic schema
            return schema.model_validate(parsed_data)

        except Exception as e:
            logger.error(f"Local LLM request failed: {e}")
            raise RuntimeError(f"Local LLM Error: {e}")
