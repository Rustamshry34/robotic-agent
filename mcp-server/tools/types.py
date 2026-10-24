from typing import Any, Optional

from pydantic import BaseModel


class ToolResult(BaseModel):
    """
    Standard response schema for all MCP robot tools.

    This structure is consumed by:
    - LangChain / DeepAgents
    - Evaluation pipeline
    - PostgreSQL logging layer
    """

    success: bool
    message: str
    data: Optional[dict[str, Any]] = None
    error_code: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
