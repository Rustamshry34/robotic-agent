"""
Application configuration for the self-verifying robotic AI agent.

This module uses environment variables only.
No .env file is loaded.

Secrets such as:
    - LLM_API_KEY
    - DATABASE_URL
    - POSTGRES_PASSWORD

should be provided via GitHub Actions secrets or deployment environment.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from functools import lru_cache
from typing import Any
from urllib.parse import quote_plus

from pydantic import (
    AliasChoices,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger(__name__)


__all__ = [
    "ChromaMode",
    "Environment",
    "LogLevel",
    "McpTransport",
    "Settings",
    "get_settings",
]


class Environment(StrEnum):
    LOCAL = "local"
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ChromaMode(StrEnum):
    PERSISTENT = "persistent"
    EPHEMERAL = "ephemeral"
    HTTP = "http"


class McpTransport(StrEnum):
    STDIO = "stdio"
    SSE = "sse"


class Settings(BaseSettings):
    """
    Central application settings.

    All values are read from environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=None,
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    app_name: str = Field(
        default="Robotic AI Agent API",
        validation_alias=AliasChoices(
            "APP_NAME",
            "PROJECT_NAME",
        ),
    )

    environment: Environment = Field(
        default=Environment.LOCAL,
        validation_alias=AliasChoices(
            "ENVIRONMENT",
            "ENV",
            "APP_ENV",
        ),
    )

    debug: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "DEBUG",
            "APP_DEBUG",
        ),
    )

    ci: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "CI",
            "GITHUB_ACTIONS",
        ),
    )

    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        validation_alias=AliasChoices(
            "LOG_LEVEL",
            "APP_LOG_LEVEL",
        ),
    )

    api_host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices(
            "API_HOST",
            "HOST",
        ),
    )

    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        validation_alias=AliasChoices(
            "API_PORT",
            "PORT",
        ),
    )

    api_workers: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices(
            "API_WORKERS",
            "WORKERS",
        ),
    )

    agent_version: str = Field(
        default="0.1.0",
        validation_alias=AliasChoices(
            "AGENT_VERSION",
            "APP_VERSION",
        ),
    )

    git_commit: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GIT_COMMIT",
            "GITHUB_SHA",
        ),
    )

    github_run_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GITHUB_RUN_ID",
        ),
    )

    github_run_number: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GITHUB_RUN_NUMBER",
        ),
    )

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    llm_provider: str = Field(
        default="openai",
        validation_alias=AliasChoices(
            "LLM_PROVIDER",
        ),
    )

    llm_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices(
            "LLM_MODEL",
            "OPENAI_MODEL",
        ),
    )

    llm_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LLM_API_KEY",
            "OPENAI_API_KEY",
        ),
    )

    llm_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LLM_BASE_URL",
            "OPENAI_API_BASE",
        ),
    )

    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        validation_alias=AliasChoices(
            "LLM_TEMPERATURE",
        ),
    )

    llm_max_tokens: int = Field(
        default=2048,
        gt=0,
        validation_alias=AliasChoices(
            "LLM_MAX_TOKENS",
        ),
    )

    llm_timeout_seconds: int = Field(
        default=60,
        gt=0,
        validation_alias=AliasChoices(
            "LLM_TIMEOUT_SECONDS",
        ),
    )

    llm_max_retries: int = Field(
        default=2,
        ge=0,
        validation_alias=AliasChoices(
            "LLM_MAX_RETRIES",
        ),
    )

    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------

    database_url: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "DATABASE_URL",
        ),
    )

    postgres_user: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "POSTGRES_USER",
        ),
    )

    postgres_password: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "POSTGRES_PASSWORD",
        ),
    )

    postgres_host: str = Field(
        default="localhost",
        validation_alias=AliasChoices(
            "POSTGRES_HOST",
        ),
    )

    postgres_port: int = Field(
        default=5432,
        ge=1,
        le=65535,
        validation_alias=AliasChoices(
            "POSTGRES_PORT",
        ),
    )

    postgres_db: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "POSTGRES_DB",
        ),
    )

    database_echo: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "DATABASE_ECHO",
        ),
    )

    database_auto_create_tables: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "DATABASE_AUTO_CREATE_TABLES",
        ),
    )

    database_pool_size: int = Field(
        default=5,
        ge=1,
        validation_alias=AliasChoices(
            "DATABASE_POOL_SIZE",
        ),
    )

    database_max_overflow: int = Field(
        default=10,
        ge=0,
        validation_alias=AliasChoices(
            "DATABASE_MAX_OVERFLOW",
        ),
    )

    database_pool_timeout: int = Field(
        default=30,
        gt=0,
        validation_alias=AliasChoices(
            "DATABASE_POOL_TIMEOUT",
        ),
    )

    database_pool_recycle: int = Field(
        default=1800,
        gt=0,
        validation_alias=AliasChoices(
            "DATABASE_POOL_RECYCLE",
        ),
    )

    database_pool_pre_ping: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "DATABASE_POOL_PRE_PING",
        ),
    )

    # ------------------------------------------------------------------
    # MCP
    # ------------------------------------------------------------------

    mcp_server_name: str = Field(
        default="robotic-ai-agent-mcp",
        validation_alias=AliasChoices(
            "MCP_SERVER_NAME",
        ),
    )

    mcp_robot_backend: str = Field(
        default="simulated",
        validation_alias=AliasChoices(
            "MCP_ROBOT_BACKEND",
        ),
    )

    mcp_transport: McpTransport = Field(
        default=McpTransport.STDIO,
        validation_alias=AliasChoices(
            "MCP_TRANSPORT",
        ),
    )

    mcp_host: str = Field(
        default="localhost",
        validation_alias=AliasChoices(
            "MCP_HOST",
        ),
    )

    mcp_port: int = Field(
        default=8001,
        ge=1,
        le=65535,
        validation_alias=AliasChoices(
            "MCP_PORT",
        ),
    )

    mcp_timeout_seconds: int = Field(
        default=30,
        gt=0,
        validation_alias=AliasChoices(
            "MCP_TIMEOUT_SECONDS",
        ),
    )

    # ------------------------------------------------------------------
    # ChromaDB
    # ------------------------------------------------------------------

    chroma_mode: ChromaMode = Field(
        default=ChromaMode.PERSISTENT,
        validation_alias=AliasChoices(
            "CHROMA_MODE",
        ),
    )

    chroma_collection: str = Field(
        default="robot_experiences",
        validation_alias=AliasChoices(
            "CHROMA_COLLECTION",
        ),
    )

    chroma_persist_dir: str = Field(
        default="./data/chroma",
        validation_alias=AliasChoices(
            "CHROMA_PERSIST_DIR",
        ),
    )

    chroma_host: str = Field(
        default="localhost",
        validation_alias=AliasChoices(
            "CHROMA_HOST",
        ),
    )

    chroma_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        validation_alias=AliasChoices(
            "CHROMA_PORT",
        ),
    )

    chroma_distance: str = Field(
        default="cosine",
        validation_alias=AliasChoices(
            "CHROMA_DISTANCE",
        ),
    )

    chroma_top_k: int = Field(
        default=5,
        ge=1,
        validation_alias=AliasChoices(
            "CHROMA_TOP_K",
        ),
    )

    # ------------------------------------------------------------------
    # Agent behavior
    # ------------------------------------------------------------------

    agent_max_plan_steps: int = Field(
        default=12,
        ge=1,
        validation_alias=AliasChoices(
            "AGENT_MAX_PLAN_STEPS",
        ),
    )

    agent_max_action_retries: int = Field(
        default=3,
        ge=0,
        validation_alias=AliasChoices(
            "AGENT_MAX_ACTION_RETRIES",
        ),
    )

    agent_enable_verification: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "AGENT_ENABLE_VERIFICATION",
        ),
    )

    agent_enable_memory: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "AGENT_ENABLE_MEMORY",
        ),
    )

    agent_task_timeout_seconds: int = Field(
        default=180,
        gt=0,
        validation_alias=AliasChoices(
            "AGENT_TASK_TIMEOUT_SECONDS",
        ),
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    evaluation_baseline: str = Field(
        default="full_system",
        validation_alias=AliasChoices(
            "EVALUATION_BASELINE",
        ),
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("environment", mode="before")
    @classmethod
    def _normalize_environment(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()

            mapping = {
                "local": Environment.LOCAL.value,
                "development": Environment.DEVELOPMENT.value,
                "dev": Environment.DEVELOPMENT.value,
                "test": Environment.TEST.value,
                "testing": Environment.TEST.value,
                "production": Environment.PRODUCTION.value,
                "prod": Environment.PRODUCTION.value,
            }

            return mapping.get(normalized, normalized)

        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().upper()

        return value

    @field_validator("chroma_mode", mode="before")
    @classmethod
    def _normalize_chroma_mode(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()

        return value

    @field_validator("mcp_transport", mode="before")
    @classmethod
    def _normalize_mcp_transport(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()

        return value

    @field_validator("mcp_robot_backend", mode="before")
    @classmethod
    def _normalize_mcp_robot_backend(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()

        return value

    @field_validator("evaluation_baseline", mode="before")
    @classmethod
    def _normalize_evaluation_baseline(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()

        return value

    @model_validator(mode="after")
    def _validate_production_settings(self) -> Settings:
        """
        Fail fast in production if critical secrets are missing.
        """
        if self.is_production:
            missing: list[str] = []

            if self.llm_api_key is None:
                missing.append("LLM_API_KEY")

            if self.build_database_url() is None:
                missing.append(
                    "DATABASE_URL or POSTGRES_USER/POSTGRES_PASSWORD/POSTGRES_DB"
                )

            if missing:
                raise ValueError(
                    "Missing required production settings: "
                    + ", ".join(missing)
                )

            if self.debug:
                logger.warning(
                    "DEBUG is enabled in production. "
                    "This is not recommended."
                )

            if self.database_auto_create_tables:
                logger.warning(
                    "DATABASE_AUTO_CREATE_TABLES is enabled in production. "
                    "Prefer Alembic migrations."
                )

        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def is_test(self) -> bool:
        return self.environment == Environment.TEST

    @property
    def logging_level(self) -> str:
        return self.log_level.value

    @staticmethod
    def _normalize_database_url(url: str) -> str:
        """
        Normalize database URL for async SQLAlchemy drivers.
        """
        url = url.strip()

        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+asyncpg://", 1)

        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)

        if url.startswith("postgresql+asyncpg://"):
            return url

        if url.startswith("sqlite://"):
            return url.replace("sqlite://", "sqlite+aiosqlite://", 1)

        if url.startswith("sqlite+aiosqlite://"):
            return url

        return url

    def build_database_url(self) -> str | None:
        """
        Build async database URL from settings.

        Priority:
            1. DATABASE_URL
            2. POSTGRES_* fields
        """
        if self.database_url is not None:
            return self._normalize_database_url(
                self.database_url.get_secret_value()
            )

        if (
            self.postgres_user
            and self.postgres_password
            and self.postgres_db
        ):
            encoded_user = quote_plus(self.postgres_user)
            encoded_password = quote_plus(
                self.postgres_password.get_secret_value()
            )

            return (
                f"postgresql+asyncpg://"
                f"{encoded_user}:{encoded_password}@"
                f"{self.postgres_host}:{self.postgres_port}/"
                f"{self.postgres_db}"
            )

        return None

    def safe_summary(self) -> dict[str, Any]:
        """
        Return a safe configuration summary without secrets.
        """
        data = self.model_dump(
            mode="json",
            exclude={
                "llm_api_key",
                "postgres_password",
                "database_url",
            },
        )

        data["llm_api_key"] = "set" if self.llm_api_key else "unset"
        data["postgres_password"] = (
            "set" if self.postgres_password else "unset"
        )
        data["database_url"] = "set" if self.database_url else "unset"

        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached settings factory.

    Usage:
        settings = get_settings()
    """
    return Settings()
