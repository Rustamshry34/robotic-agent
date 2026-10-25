"""
ChromaDB vector memory client for the self-verifying robotic AI agent.

This module stores and retrieves experience records used for:
- task success memory
- failure memory
- post-error recovery memory
- state-aware experience retrieval

Configuration is read from environment variables. No .env file is required.

Environment variables:
    CHROMA_MODE:
        persistent | ephemeral | http
        Default: persistent

    CHROMA_PERSIST_DIR:
        Local persistence directory for ChromaDB.
        Default: ./data/chroma

    CHROMA_HOST:
        Chroma HTTP host if CHROMA_MODE=http.
        Default: localhost

    CHROMA_PORT:
        Chroma HTTP port if CHROMA_MODE=http.
        Default: 8000

    CHROMA_COLLECTION:
        Chroma collection name.
        Default: robot_experiences

    CHROMA_DISTANCE:
        Distance metric for vector search.
        Default: cosine
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb


logger = logging.getLogger(__name__)


DEFAULT_COLLECTION_NAME = "robot_experiences"
DEFAULT_PERSIST_DIR = "./data/chroma"
DEFAULT_MODE = "persistent"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
DEFAULT_DISTANCE = "cosine"

MAX_DOCUMENT_CHARS = 8000
MAX_METADATA_TEXT_CHARS = 2000


class ChromaMemoryClient:
    """
    Client for storing and retrieving robot experience memory in ChromaDB.

    The memory is intentionally designed for LLMOps observability:
    - structured metadata for filtering
    - semantic document text for similarity search
    - deterministic IDs for reproducibility
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        mode: str = DEFAULT_MODE,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        distance_metric: str = DEFAULT_DISTANCE,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.mode = mode.lower().strip()
        self.host = host
        self.port = int(port)
        self.distance_metric = distance_metric

        self._client = self._create_client()
        self._collection = self._get_or_create_collection()

        logger.info(
            "ChromaMemoryClient initialized. collection=%s mode=%s",
            self.collection_name,
            self.mode,
        )

    @classmethod
    def from_env(cls) -> ChromaMemoryClient:
        """
        Create a ChromaMemoryClient from environment variables.
        """
        return cls(
            collection_name=os.getenv("CHROMA_COLLECTION", DEFAULT_COLLECTION_NAME),
            persist_directory=os.getenv("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR),
            mode=os.getenv("CHROMA_MODE", DEFAULT_MODE),
            host=os.getenv("CHROMA_HOST", DEFAULT_HOST),
            port=int(os.getenv("CHROMA_PORT", str(DEFAULT_PORT))),
            distance_metric=os.getenv("CHROMA_DISTANCE", DEFAULT_DISTANCE),
        )

    def _create_client(self) -> chromadb.api.Client:
        """
        Create the appropriate Chroma client based on CHROMA_MODE.
        """
        if self.mode == "ephemeral":
            logger.info("Using ephemeral ChromaDB client.")
            return chromadb.EphemeralClient()

        if self.mode == "http":
            logger.info(
                "Using HTTP ChromaDB client. host=%s port=%s",
                self.host,
                self.port,
            )
            return chromadb.HttpClient(
                host=self.host,
                port=self.port,
            )

        if self.mode == "persistent":
            persist_path = Path(self.persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Using persistent ChromaDB client. path=%s",
                persist_path.resolve(),
            )

            return chromadb.PersistentClient(path=str(persist_path))

        raise ValueError(
            f"Unknown CHROMA_MODE: '{self.mode}'. "
            "Expected one of: persistent, ephemeral, http."
        )

    def _get_or_create_collection(self) -> chromadb.api.models.Collection.Collection:
        """
        Get or create the Chroma collection.
        """
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": self.distance_metric,
            },
        )

    @staticmethod
    def _normalize_state_summary(
        state_summary: str | dict[str, Any] | None,
    ) -> str:
        """
        Convert state summary into a stable string representation.
        """
        if state_summary is None:
            return ""

        if isinstance(state_summary, str):
            text = state_summary.strip()
        else:
            text = json.dumps(
                state_summary,
                ensure_ascii=False,
                default=str,
                sort_keys=True,
            )

        if len(text) > MAX_METADATA_TEXT_CHARS:
            text = text[:MAX_METADATA_TEXT_CHARS]

        return text

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Chroma metadata values must be scalar or JSON-serializable strings.

        This method prevents invalid metadata types from breaking ChromaDB.
        """
        sanitized: dict[str, Any] = {}

        for key, value in metadata.items():
            if value is None:
                sanitized[key] = "unknown"
            elif isinstance(value, bool):
                sanitized[key] = value
            elif isinstance(value, (int, float, str)):
                sanitized[key] = value
            else:
                sanitized[key] = json.dumps(
                    value,
                    ensure_ascii=False,
                    default=str,
                )

        return sanitized

    @staticmethod
    def _build_where(
        clauses: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Build a ChromaDB where filter from multiple clauses.
        """
        clauses = [clause for clause in clauses if clause]

        if not clauses:
            return None

        if len(clauses) == 1:
            return clauses[0]

        return {"$and": clauses}

    @staticmethod
    def build_experience_document(
        action: str,
        outcome: str,
        state_summary: str = "",
        error_code: str | None = None,
        error_message: str | None = None,
        resolution: str | None = None,
        task_name: str | None = None,
    ) -> str:
        """
        Build the semantic document stored in ChromaDB.

        This document is used for vector similarity search.
        """
        parts = [
            f"Action: {action}",
            f"Outcome: {outcome}",
        ]

        if task_name:
            parts.append(f"Task: {task_name}")

        if state_summary:
            parts.append(f"State: {state_summary}")

        if error_code and error_code.upper() != "NONE":
            parts.append(f"Error code: {error_code}")

        if error_message:
            parts.append(f"Error message: {error_message}")

        if resolution:
            parts.append(f"Resolution: {resolution}")

        document = "\n".join(parts)

        if len(document) > MAX_DOCUMENT_CHARS:
            document = document[:MAX_DOCUMENT_CHARS]

        return document

    def add_experience(
        self,
        action: str,
        outcome: str,
        state_summary: str | dict[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        resolution: str | None = None,
        task_id: str | None = None,
        task_name: str | None = None,
        step_index: int | None = None,
        metadata: dict[str, Any] | None = None,
        experience_id: str | None = None,
    ) -> str:
        """
        Store a single experience in ChromaDB.

        Args:
            action:
                MCP tool/action name.
                Example: "grasp_object".

            outcome:
                Expected values:
                - success
                - failure
                - recovered

            state_summary:
                Human-readable or structured robot/environment state.

            error_code:
                Machine-readable error code.
                Example: "GRASP_FAILED".

            error_message:
                Detailed error message.

            resolution:
                Recovery or mitigation strategy.

            task_id:
                Unique task execution ID from PostgreSQL.

            task_name:
                High-level task name.
                Example: "take_red_cup_to_kitchen_counter".

            step_index:
                Step number inside the task plan.

            metadata:
                Additional metadata.

            experience_id:
                Optional deterministic ID for reproducibility.

        Returns:
            The stored experience ID.
        """
        experience_id = experience_id or str(uuid.uuid4())

        action_normalized = action.strip().lower()
        outcome_normalized = outcome.strip().lower()
        error_code_normalized = (
            error_code.strip().upper()
            if error_code
            else "NONE"
        )

        state_text = self._normalize_state_summary(state_summary)

        document = self.build_experience_document(
            action=action_normalized,
            outcome=outcome_normalized,
            state_summary=state_text,
            error_code=error_code_normalized,
            error_message=error_message,
            resolution=resolution,
            task_name=task_name,
        )

        base_metadata: dict[str, Any] = {
            "experience_id": experience_id,
            "action": action_normalized,
            "outcome": outcome_normalized,
            "error_code": error_code_normalized,
            "task_id": task_id or "unknown",
            "task_name": task_name or "unknown",
            "step_index": int(step_index) if step_index is not None else -1,
            "state_summary": state_text,
            "created_at": datetime.now(UTC).isoformat(),
        }

        if error_message:
            base_metadata["error_message"] = error_message.strip()

        if resolution:
            base_metadata["resolution"] = resolution.strip()

        if metadata:
            base_metadata.update(metadata)

        sanitized_metadata = self._sanitize_metadata(base_metadata)

        self._collection.add(
            documents=[document],
            metadatas=[sanitized_metadata],
            ids=[experience_id],
        )

        logger.info(
            "Experience stored. id=%s action=%s outcome=%s error_code=%s",
            experience_id,
            action_normalized,
            outcome_normalized,
            error_code_normalized,
        )

        return experience_id

    def add_experiences(
        self,
        experiences: Iterable[dict[str, Any]],
    ) -> list[str]:
        """
        Store multiple experiences in one batch.

        Each item should match the keyword arguments of add_experience().
        """
        experience_ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []

        for experience in experiences:
            experience_id = experience.get("experience_id") or str(uuid.uuid4())

            action = experience.get("action", "")
            outcome = experience.get("outcome", "")

            if not action or not outcome:
                logger.warning(
                    "Skipping experience because action or outcome is missing. "
                    "experience=%s",
                    experience,
                )
                continue

            action_normalized = action.strip().lower()
            outcome_normalized = outcome.strip().lower()
            error_code_normalized = (
                experience.get("error_code", "NONE") or "NONE"
            ).strip().upper()

            state_text = self._normalize_state_summary(
                experience.get("state_summary")
            )

            document = self.build_experience_document(
                action=action_normalized,
                outcome=outcome_normalized,
                state_summary=state_text,
                error_code=error_code_normalized,
                error_message=experience.get("error_message"),
                resolution=experience.get("resolution"),
                task_name=experience.get("task_name"),
            )

            base_metadata: dict[str, Any] = {
                "experience_id": experience_id,
                "action": action_normalized,
                "outcome": outcome_normalized,
                "error_code": error_code_normalized,
                "task_id": experience.get("task_id") or "unknown",
                "task_name": experience.get("task_name") or "unknown",
                "step_index": (
                    int(experience["step_index"])
                    if experience.get("step_index") is not None
                    else -1
                ),
                "state_summary": state_text,
                "created_at": datetime.now(UTC).isoformat(),
            }

            if experience.get("error_message"):
                base_metadata["error_message"] = experience["error_message"].strip()

            if experience.get("resolution"):
                base_metadata["resolution"] = experience["resolution"].strip()

            extra_metadata = experience.get("metadata") or {}
            if extra_metadata:
                base_metadata.update(extra_metadata)

            sanitized_metadata = self._sanitize_metadata(base_metadata)

            experience_ids.append(experience_id)
            documents.append(document)
            metadatas.append(sanitized_metadata)
            ids.append(experience_id)

        if ids:
            self._collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

            logger.info("Stored %s experiences in batch.", len(ids))

        return experience_ids

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query ChromaDB for relevant experiences.
        """
        query_text = query_text.strip()

        if not query_text:
            query_text = "robot experience"

        if n_results <= 0:
            return []

        query_kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }

        if where:
            query_kwargs["where"] = where

        raw = self._collection.query(**query_kwargs)

        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        results: list[dict[str, Any]] = []

        for index, experience_id in enumerate(ids):
            results.append(
                {
                    "id": experience_id,
                    "document": documents[index] if index < len(documents) else None,
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                    "distance": distances[index] if index < len(distances) else None,
                }
            )

        logger.debug(
            "Chroma query returned %s results. query=%s",
            len(results),
            query_text,
        )

        return results

    def retrieve_relevant_experiences(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generic retrieval method for agent context building.
        """
        return self.query(
            query_text=query_text,
            n_results=n_results,
            where=where,
        )

    def retrieve_error_recovery_experiences(
        self,
        action: str,
        error_code: str,
        state_summary: str | dict[str, Any] | None = None,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve experiences relevant to recovering from a specific error.

        Retrieval strategy:
        1. Filter by action + error_code
        2. Fallback to error_code only
        3. Fallback to action only
        4. Fallback to pure semantic search
        """
        action_normalized = action.strip().lower() if action else ""
        error_code_normalized = (
            error_code.strip().upper()
            if error_code
            else "NONE"
        )

        state_text = self._normalize_state_summary(state_summary)

        query_parts = []

        if action_normalized:
            query_parts.append(f"Action: {action_normalized}")

        if error_code_normalized != "NONE":
            query_parts.append(f"Error code: {error_code_normalized}")

        if state_text:
            query_parts.append(f"State: {state_text}")

        query_text = "\n".join(query_parts) or "robot error recovery experience"

        clauses: list[dict[str, Any]] = []

        if action_normalized:
            clauses.append({"action": action_normalized})

        if error_code_normalized != "NONE":
            clauses.append({"error_code": error_code_normalized})

        where = self._build_where(clauses)

        results = self.query(
            query_text=query_text,
            n_results=n_results,
            where=where,
        )

        if results:
            return results

        if error_code_normalized != "NONE":
            results = self.query(
                query_text=query_text,
                n_results=n_results,
                where={"error_code": error_code_normalized},
            )

            if results:
                return results

        if action_normalized:
            results = self.query(
                query_text=query_text,
                n_results=n_results,
                where={"action": action_normalized},
            )

            if results:
                return results

        return self.query(
            query_text=query_text,
            n_results=n_results,
        )

    def retrieve_success_experiences(
        self,
        action: str,
        state_summary: str | dict[str, Any] | None = None,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve successful experiences for a given action.
        """
        action_normalized = action.strip().lower() if action else ""
        state_text = self._normalize_state_summary(state_summary)

        query_parts = [
            f"Action: {action_normalized}",
            "Outcome: success",
        ]

        if state_text:
            query_parts.append(f"State: {state_text}")

        query_text = "\n".join(query_parts)

        where = self._build_where(
            [
                {"action": action_normalized},
                {"outcome": "success"},
            ]
        )

        return self.query(
            query_text=query_text,
            n_results=n_results,
            where=where,
        )

    def count(self) -> int:
        """
        Return the number of experiences in the collection.
        """
        return self._collection.count()

    def info(self) -> dict[str, Any]:
        """
        Return basic memory client information.
        """
        return {
            "collection_name": self.collection_name,
            "mode": self.mode,
            "persist_directory": self.persist_directory,
            "distance_metric": self.distance_metric,
            "experience_count": self.count(),
        }

    def delete_collection(self) -> None:
        """
        Delete and recreate the collection.

        Intended for tests and evaluation resets.
        """
        try:
            self._client.delete_collection(name=self.collection_name)
            logger.info(
                "Deleted Chroma collection: %s",
                self.collection_name,
            )
        except Exception as exc:
            logger.debug(
                "Could not delete collection %s: %s",
                self.collection_name,
                exc,
            )

        self._collection = self._get_or_create_collection()


@lru_cache(maxsize=1)
def get_chroma_memory() -> ChromaMemoryClient:
    """
    Singleton factory for FastAPI dependency injection.

    Usage:
        memory = get_chroma_memory()
    """
    return ChromaMemoryClient.from_env()
