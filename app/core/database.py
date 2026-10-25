"""
Async database engine, session factory, and FastAPI dependencies.

This module is designed for:
- PostgreSQL + asyncpg in production
- GitHub Actions environment variables / secrets
- FastAPI dependency injection
- testable and reproducible database lifecycle

Important:
    No .env file is used.
    Database configuration must be provided via environment variables.

Primary environment variable:
    DATABASE_URL

Alternative PostgreSQL environment variables:
    POSTGRES_USER
    POSTGRES_PASSWORD
    POSTGRES_HOST
    POSTGRES_PORT
    POSTGRES_DB
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from functools import lru_cache
from urllib.parse import quote_plus, urlparse, urlunparse

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from memory.models import Base

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


__all__ = [
    "check_database_connection",
    "dispose_db",
    "get_db_session",
    "get_engine",
    "get_session_factory",
    "init_db",
    "shutdown_db",
    "startup_db",
]


def _env_bool(name: str, default: bool = False) -> bool:
    """
    Read a boolean environment variable.
    """
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    return raw_value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_int(name: str, default: int) -> int:
    """
    Read an integer environment variable.
    """
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError:
        logger.warning(
            "Invalid integer environment variable %s=%s. Using default=%s.",
            name,
            raw_value,
            default,
        )
        return default


def _normalize_database_url(url: str) -> str:
    """
    Normalize database URL for SQLAlchemy async drivers.

    Examples:
        postgres://... -> postgresql+asyncpg://...
        postgresql://... -> postgresql+asyncpg://...
        sqlite://... -> sqlite+aiosqlite://...
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


def _build_database_url_from_parts() -> str:
    """
    Build PostgreSQL URL from individual environment variables.

    This is useful for GitHub Actions service containers where
    POSTGRES_* variables are provided.
    """
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB")

    if not user or not password or not database:
        raise RuntimeError(
            "Database configuration is missing. "
            "Set DATABASE_URL or POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB."
        )

    encoded_user = quote_plus(user)
    encoded_password = quote_plus(password)

    return (
        f"postgresql+asyncpg://"
        f"{encoded_user}:{encoded_password}@"
        f"{host}:{port}/"
        f"{database}"
    )


def _get_database_url() -> str:
    """
    Resolve database URL from environment.

    Priority:
        1. DATABASE_URL
        2. POSTGRES_* variables
    """
    database_url = os.getenv("DATABASE_URL")

    if database_url:
        return _normalize_database_url(database_url)

    return _build_database_url_from_parts()


def _redact_url(url: str) -> str:
    """
    Redact password from database URL for safe logging.
    """
    try:
        parsed = urlparse(url)

        if not parsed.password:
            return url

        netloc = parsed.netloc.replace(
            parsed.password,
            "***",
        )

        return urlunparse(
            (
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )
    except Exception:
        return "<database-url>"


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """
    Create and cache the async SQLAlchemy engine.
    """
    database_url = _get_database_url()

    engine_kwargs: dict = {
        "echo": _env_bool("DATABASE_ECHO", default=False),
    }

    connect_args: dict = {}

    if database_url.startswith("postgresql"):
        engine_kwargs.update(
            {
                "pool_size": _env_int("DATABASE_POOL_SIZE", default=5),
                "max_overflow": _env_int("DATABASE_MAX_OVERFLOW", default=10),
                "pool_timeout": _env_int("DATABASE_POOL_TIMEOUT", default=30),
                "pool_recycle": _env_int("DATABASE_POOL_RECYCLE", default=1800),
                "pool_pre_ping": _env_bool("DATABASE_POOL_PRE_PING", default=True),
            }
        )

        connect_args["server_settings"] = {
            "application_name": "robotic-ai-agent",
        }

    if connect_args:
        engine_kwargs["connect_args"] = connect_args

    logger.info(
        "Creating async database engine. url=%s",
        _redact_url(database_url),
    )

    return create_async_engine(
        database_url,
        **engine_kwargs,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker:
    """
    Create and cache the async session factory.
    """
    return async_sessionmaker(
        bind=get_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.

    Usage:
        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db_session)):
            ...

    Behavior:
        - commits automatically if the endpoint succeeds
        - rolls back automatically if the endpoint raises
    """
    session_factory = get_session_factory()

    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """
    Create database tables if enabled.

    For production, prefer Alembic migrations.
    Set DATABASE_AUTO_CREATE_TABLES=false in production.
    """
    auto_create_tables = _env_bool(
        "DATABASE_AUTO_CREATE_TABLES",
        default=True,
    )

    if not auto_create_tables:
        logger.info(
            "DATABASE_AUTO_CREATE_TABLES is disabled. "
            "Skipping Base.metadata.create_all."
        )
        return

    logger.info("Creating database tables if they do not exist.")

    engine = get_engine()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    logger.info("Database tables ensured.")


async def check_database_connection() -> bool:
    """
    Check database connectivity.

    Returns:
        True if SELECT 1 succeeds, otherwise False.
    """
    try:
        engine = get_engine()

        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

        logger.info("Database connection check succeeded.")
        return True

    except Exception:
        logger.exception("Database connection check failed.")
        return False


async def startup_db() -> None:
    """
    Database startup hook for FastAPI lifespan.

    Usage:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await startup_db()
            yield
            await shutdown_db()
    """
    healthy = await check_database_connection()

    if not healthy:
        raise RuntimeError(
            "Database connection failed. "
            "Check DATABASE_URL or POSTGRES_* environment variables."
        )

    await init_db()


async def dispose_db() -> None:
    """
    Dispose the SQLAlchemy engine and clear caches.
    """
    engine = get_engine()

    await engine.dispose()

    get_engine.cache_clear()
    get_session_factory.cache_clear()

    logger.info("Database engine disposed.")


async def shutdown_db() -> None:
    """
    Database shutdown hook for FastAPI lifespan.
    """
    await dispose_db()
