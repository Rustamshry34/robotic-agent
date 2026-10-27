"""
Custom Chat Model for OpenRouter with native reasoning support.
"""

from __future__ import annotations

import json
import logging
from typing import Any, List, Optional, Type

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda, RunnableSequence
from pydantic import BaseModel, Field, SecretStr

logger = logging.getLogger(__name__)


def _convert_messages_to_openrouter_format(messages: List[BaseMessage]) -> List[dict]:
    """Convert LangChain messages to OpenRouter format, preserving reasoning_details."""
    formatted = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            formatted.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            formatted.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            msg_dict = {"role": "assistant", "content": msg.content}
            # Preserve reasoning_details from previous turns if it exists
            if msg.additional_kwargs.get("reasoning_details"):
                msg_dict["reasoning_details"] = msg.additional_kwargs["reasoning_details"]
            formatted.append(msg_dict)
        elif isinstance(msg, ToolMessage):
            formatted.append({
                "role": "tool", 
                "content": msg.content, 
                "tool_call_id": msg.tool_call_id
            })
    return formatted


class ChatOpenRouter(BaseChatModel):
    """
    Custom LangChain Chat Model for OpenRouter.
    Supports native 'reasoning' parameter and preserves 'reasoning_details'.
    """
    api_key: SecretStr = Field(..., description="OpenRouter API Key")
    model: str = Field(default="google/gemma-4-31b-it:free")
    base_url: str = Field(default="https://openrouter.ai/api/v1/chat/completions")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=4096)
    timeout: int = Field(default=60)
    reasoning_enabled: bool = Field(default=True)

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": _convert_messages_to_openrouter_format(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # OpenRouter specific: reasoning must be at the root level
        if self.reasoning_enabled:
            payload["reasoning"] = {"enabled": True}

        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
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

            choice = data["choices"][0]["message"]
            content = choice.get("content", "")
            
            additional_kwargs = {}
            if "reasoning_details" in choice:
                additional_kwargs["reasoning_details"] = choice["reasoning_details"]

            generation_info = {
                "finish_reason": data["choices"][0].get("finish_reason"),
                "model": data.get("model"),
            }

            message = AIMessage(
                content=content,
                additional_kwargs=additional_kwargs,
            )

            return ChatResult(
                generations=[ChatGeneration(message=message, generation_info=generation_info)]
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter API HTTP Error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"OpenRouter request failed: {e}")
            raise

    def _generate(self, *args, **kwargs):
        """Sync fallback if ever needed, though we primarily use async."""
        import asyncio
        return asyncio.run(self._agenerate(*args, **kwargs))

    def with_structured_output(
        self,
        schema: Type[BaseModel],
        *,
        method: str = "json_schema",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        """
        Return a runnable that outputs structured data matching the schema.
        
        This method enables LangChain's structured output pattern by:
        1. Injecting JSON schema into system message
        2. Using JsonOutputParser to parse model response
        3. Returning a RunnableSequence that can be chained with prompts
        
        Args:
            schema: Pydantic model class to parse output into
            method: Output method (currently only 'json_schema' supported)
            include_raw: Whether to include raw model output
            **kwargs: Additional arguments
            
        Returns:
            Runnable that outputs validated Pydantic model instances
        """
        if method != "json_schema":
            raise ValueError(f"Only 'json_schema' method is supported, got '{method}'")
        
        # Generate JSON schema from Pydantic model
        json_schema = schema.model_json_schema()
        
        # Create system message with schema
        system_message = SystemMessage(
            content=f"""You must respond with a valid JSON object that strictly follows this schema:

{json.dumps(json_schema, indent=2)}

IMPORTANT RULES:
1. Output ONLY the JSON object, no additional text, explanations, or markdown formatting
2. All required fields must be present
3. Field types must match the schema exactly
4. Do not include ```json or ``` markers"""
        )
        
        # Create output parser
        parser = JsonOutputParser(pydantic_object=schema)
        
        # Create runnable that prepends system message
        async def _prepend_system_prompt(messages: List[BaseMessage]) -> List[BaseMessage]:
            return [system_message] + list(messages)
        
        # Build the chain: prepend system prompt -> call model -> parse output
        chain = RunnableSequence(
            first=RunnableLambda(_prepend_system_prompt),
            last=RunnableSequence(
                first=self,
                last=parser,
            ),
        )
        
        return chain
