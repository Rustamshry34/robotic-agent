"""
Pure Async OpenRouter Client.
No LangChain dependencies. Direct HTTP control with native reasoning support.
"""

import json
import logging
from typing import Type

import httpx
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """
    Lightweight, async HTTP client for OpenRouter API.
    Enforces strict JSON schema output and supports native reasoning.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "google/gemma-4-31b-it:free",
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 60,
        reasoning_enabled: bool = True,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.reasoning_enabled = reasoning_enabled

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
2. Do NOT include any additional text, explanations, or markdown formatting (NO ```json or ``` markers).
3. All required fields must be present.
4. Field types must match the schema exactly.
"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": full_system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # OpenRouter native reasoning support (must be at root level)
        if self.reasoning_enabled:
            payload["reasoning"] = {"enabled": True}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/robotic-agent",
            "X-Title": "Self-Verifying Robotic AI Agent",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            content = data["choices"][0]["message"].get("content", "").strip()
            
            # Fallback: Clean markdown code blocks if the model ignores instructions
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()

            # 1. Parse JSON
            parsed_data = json.loads(content)
            
            # 2. Validate against Pydantic schema
            return schema.model_validate(parsed_data)

        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter API HTTP Error: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"OpenRouter API Error: {e.response.status_code}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM. Raw output: {content}")
            raise RuntimeError(f"Invalid JSON response from LLM: {e}")
        except ValidationError as e:
            logger.error(f"Pydantic validation failed for LLM response: {e}")
            raise RuntimeError(f"LLM response did not match schema: {e}")
        except Exception as e:
            logger.error(f"OpenRouter request failed unexpectedly: {e}")
            raise
