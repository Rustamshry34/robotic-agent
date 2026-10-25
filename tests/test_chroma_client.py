"""
Tests for the ChromaDB vector memory client.

This module tests:
- Client initialization (ephemeral, persistent, http modes)
- Experience storage (single and batch)
- Query and retrieval methods
- Error recovery retrieval with fallback strategy
- Success experience retrieval
- Metadata sanitization
- Where clause building
- Helper functions
- Collection management

All tests use ephemeral ChromaDB to avoid file system pollution.
"""

from __future__ import annotations

import os
import uuid

import pytest

from memory.chroma_client import (
    ChromaMemoryClient,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DISTANCE,
    DEFAULT_MODE,
    DEFAULT_PERSIST_DIR,
    MAX_DOCUMENT_CHARS,
    MAX_METADATA_TEXT_CHARS,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def ephemeral_mode(monkeypatch):
    """
    Force ephemeral mode for all tests.

    This ensures tests don't pollute the file system and
    each test gets a clean in-memory ChromaDB instance.
    """
    monkeypatch.setenv("CHROMA_MODE", "ephemeral")
    monkeypatch.setenv("CHROMA_COLLECTION", "test_robot_experiences")


@pytest.fixture
def client():
    """
    Create a fresh ephemeral ChromaMemoryClient.

    Each test gets a new client with a unique collection name
    to ensure complete isolation between tests.
    """
    unique_collection = f"test_{uuid.uuid4().hex[:8]}"
    return ChromaMemoryClient(
        mode="ephemeral",
        collection_name=unique_collection,
    )


@pytest.fixture
def sample_experience():
    """Create a sample experience dict for testing."""
    return {
        "action": "robot_grasp_object",
        "outcome": "success",
        "state_summary": {
            "robot_location": "table",
            "gripper_open": False,
            "holding_object": "red_cup",
        },
        "task_id": "test-task-123",
        "task_name": "move_red_cup",
        "step_index": 3,
    }


@pytest.fixture
def sample_failure_experience():
    """Create a sample failure experience dict."""
    return {
        "action": "robot_grasp_object",
        "outcome": "failure",
        "state_summary": {
            "robot_location": "table",
            "gripper_open": True,
            "holding_object": None,
        },
        "error_code": "GRASP_FAILED",
        "error_message": "Gripper misalignment detected.",
        "resolution": "Reposition and retry.",
        "task_id": "test-task-456",
        "task_name": "move_red_cup",
        "step_index": 3,
    }


# ----------------------------------------------------------------------
# TestClientInitialization
# ----------------------------------------------------------------------


class TestClientInitialization:
    def test_create_ephemeral_client(self):
        """Test creating an ephemeral client."""
        client = ChromaMemoryClient(mode="ephemeral")
        assert client.mode == "ephemeral"
        assert client.count() == 0

    def test_create_client_with_custom_collection(self):
        """Test creating a client with custom collection name."""
        client = ChromaMemoryClient(
            mode="ephemeral",
            collection_name="custom_collection",
        )
        assert client.collection_name == "custom_collection"

    def test_create_client_with_custom_distance(self):
        """Test creating a client with custom distance metric."""
        client = ChromaMemoryClient(
            mode="ephemeral",
            distance_metric="l2",
        )
        assert client.distance_metric == "l2"

    def test_from_env_uses_environment_variables(self, monkeypatch):
        """Test that from_env reads environment variables."""
        monkeypatch.setenv("CHROMA_MODE", "ephemeral")
        monkeypatch.setenv("CHROMA_COLLECTION", "env_collection")
        monkeypatch.setenv("CHROMA_DISTANCE", "cosine")

        client = ChromaMemoryClient.from_env()

        assert client.mode == "ephemeral"
        assert client.collection_name == "env_collection"
        assert client.distance_metric == "cosine"

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown CHROMA_MODE"):
            ChromaMemoryClient(mode="invalid_mode")

    def test_info_returns_client_metadata(self, client):
        """Test that info() returns client metadata."""
        info = client.info()

        assert "collection_name" in info
        assert "mode" in info
        assert "distance_metric" in info
        assert "experience_count" in info
        assert info["experience_count"] == 0


# ----------------------------------------------------------------------
# TestAddExperience
# ----------------------------------------------------------------------


class TestAddExperience:
    def test_add_single_experience(self, client, sample_experience):
        """Test adding a single experience."""
        experience_id = client.add_experience(**sample_experience)

        assert experience_id is not None
        assert isinstance(experience_id, str)
        assert client.count() == 1

    def test_add_experience_with_custom_id(self, client, sample_experience):
        """Test adding an experience with a custom ID."""
        custom_id = "custom-exp-id"
        experience_id = client.add_experience(
            experience_id=custom_id,
            **sample_experience,
        )

        assert experience_id == custom_id

    def test_add_experience_normalizes_action(self, client, sample_experience):
        """Test that action is normalized to lowercase."""
        sample_experience["action"] = "ROBOT_GRASP_OBJECT"
        client.add_experience(**sample_experience)

        results = client.query("grasp", n_results=1)
        assert len(results) == 1
        assert results[0]["metadata"]["action"] == "robot_grasp_object"

    def test_add_experience_normalizes_outcome(self, client, sample_experience):
        """Test that outcome is normalized to lowercase."""
        sample_experience["outcome"] = "SUCCESS"
        client.add_experience(**sample_experience)

        results = client.query("success", n_results=1)
        assert len(results) == 1
        assert results[0]["metadata"]["outcome"] == "success"

    def test_add_experience_normalizes_error_code(self, client, sample_failure_experience):
        """Test that error_code is normalized to uppercase."""
        sample_failure_experience["error_code"] = "grasp_failed"
        client.add_experience(**sample_failure_experience)

        results = client.query("grasp failed", n_results=1)
        assert len(results) == 1
        assert results[0]["metadata"]["error_code"] == "GRASP_FAILED"

    def test_add_experience_with_dict_state_summary(self, client, sample_experience):
        """Test adding experience with dict state summary."""
        client.add_experience(**sample_experience)

        results = client.query("grasp", n_results=1)
        assert len(results) == 1
        assert "state_summary" in results[0]["metadata"]

    def test_add_experience_with_string_state_summary(self, client):
        """Test adding experience with string state summary."""
        client.add_experience(
            action="robot_move_to",
            outcome="success",
            state_summary="Robot is at table",
        )

        results = client.query("move", n_results=1)
        assert len(results) == 1

    def test_add_experience_with_none_state_summary(self, client):
        """Test adding experience with None state summary."""
        client.add_experience(
            action="robot_get_state",
            outcome="success",
            state_summary=None,
        )

        assert client.count() == 1

    def test_add_experience_without_error_code(self, client):
        """Test adding experience without error code defaults to NONE."""
        client.add_experience(
            action="robot_get_state",
            outcome="success",
        )

        results = client.query("state", n_results=1)
        assert results[0]["metadata"]["error_code"] == "NONE"

    def test_add_experience_truncates_long_state_summary(self, client):
        """Test that long state summaries are truncated."""
        long_summary = "x" * (MAX_METADATA_TEXT_CHARS + 1000)
        client.add_experience(
            action="robot_get_state",
            outcome="success",
            state_summary=long_summary,
        )

        results = client.query("state", n_results=1)
        stored_summary = results[0]["metadata"]["state_summary"]
        assert len(stored_summary) <= MAX_METADATA_TEXT_CHARS


# ----------------------------------------------------------------------
# TestAddExperiences
# ----------------------------------------------------------------------


class TestAddExperiences:
    def test_add_multiple_experiences(self, client):
        """Test adding multiple experiences in batch."""
        experiences = [
            {
                "action": "robot_get_state",
                "outcome": "success",
            },
            {
                "action": "robot_move_to",
                "outcome": "success",
            },
            {
                "action": "robot_grasp_object",
                "outcome": "failure",
                "error_code": "GRASP_FAILED",
            },
        ]

        ids = client.add_experiences(experiences)

        assert len(ids) == 3
        assert client.count() == 3

    def test_add_experiences_skips_invalid(self, client):
        """Test that invalid experiences are skipped."""
        experiences = [
            {
                "action": "robot_get_state",
                "outcome": "success",
            },
            {
                "action": "",  # Invalid: empty action
                "outcome": "success",
            },
            {
                "action": "robot_move_to",
                "outcome": "",  # Invalid: empty outcome
            },
        ]

        ids = client.add_experiences(experiences)

        assert len(ids) == 1
        assert client.count() == 1

    def test_add_experiences_empty_list(self, client):
        """Test adding empty list of experiences."""
        ids = client.add_experiences([])
        assert ids == []
        assert client.count() == 0


# ----------------------------------------------------------------------
# TestQuery
# ----------------------------------------------------------------------


class TestQuery:
    def test_query_returns_results(self, client, sample_experience):
        """Test that query returns results."""
        client.add_experience(**sample_experience)

        results = client.query("grasp object", n_results=5)

        assert len(results) > 0
        assert "id" in results[0]
        assert "document" in results[0]
        assert "metadata" in results[0]
        assert "distance" in results[0]

    def test_query_respects_n_results(self, client):
        """Test that query respects n_results parameter."""
        for i in range(10):
            client.add_experience(
                action=f"robot_action_{i}",
                outcome="success",
            )

        results = client.query("action", n_results=3)
        assert len(results) == 3

    def test_query_with_empty_text(self, client, sample_experience):
        """Test that query with empty text uses default."""
        client.add_experience(**sample_experience)

        results = client.query("", n_results=1)
        assert len(results) == 1

    def test_query_with_zero_n_results(self, client):
        """Test that query with n_results=0 returns empty list."""
        results = client.query("test", n_results=0)
        assert results == []

    def test_query_with_where_filter(self, client):
        """Test query with where filter."""
        client.add_experience(
            action="robot_grasp_object",
            outcome="success",
        )
        client.add_experience(
            action="robot_grasp_object",
            outcome="failure",
            error_code="GRASP_FAILED",
        )

        results = client.query(
            "grasp",
            n_results=5,
            where={"outcome": "success"},
        )

        assert len(results) == 1
        assert results[0]["metadata"]["outcome"] == "success"


# ----------------------------------------------------------------------
# TestRetrieveRelevantExperiences
# ----------------------------------------------------------------------


class TestRetrieveRelevantExperiences:
    def test_retrieve_relevant_experiences(self, client, sample_experience):
        """Test retrieving relevant experiences."""
        client.add_experience(**sample_experience)

        results = client.retrieve_relevant_experiences(
            query_text="grasp red cup",
            n_results=5,
        )

        assert len(results) > 0

    def test_retrieve_with_where_filter(self, client):
        """Test retrieving with where filter."""
        client.add_experience(
            action="robot_grasp_object",
            outcome="success",
        )
        client.add_experience(
            action="robot_move_to",
            outcome="success",
        )

        results = client.retrieve_relevant_experiences(
            query_text="robot action",
            n_results=5,
            where={"action": "robot_grasp_object"},
        )

        assert len(results) == 1
        assert results[0]["metadata"]["action"] == "robot_grasp_object"


# ----------------------------------------------------------------------
# TestRetrieveErrorRecoveryExperiences
# ----------------------------------------------------------------------


class TestRetrieveErrorRecoveryExperiences:
    def test_retrieve_error_recovery_exact_match(self, client, sample_failure_experience):
        """Test retrieving error recovery experiences with exact match."""
        client.add_experience(**sample_failure_experience)

        results = client.retrieve_error_recovery_experiences(
            action="robot_grasp_object",
            error_code="GRASP_FAILED",
            n_results=5,
        )

        assert len(results) > 0
        assert results[0]["metadata"]["action"] == "robot_grasp_object"
        assert results[0]["metadata"]["error_code"] == "GRASP_FAILED"

    def test_retrieve_error_recovery_fallback_to_error_code(self, client):
        """Test fallback to error_code only when action+error_code fails."""
        # Add experience with different action but same error code
        client.add_experience(
            action="robot_release_object",
            outcome="failure",
            error_code="GRASP_FAILED",
            resolution="Retry with different approach",
        )

        results = client.retrieve_error_recovery_experiences(
            action="robot_grasp_object",  # Different action
            error_code="GRASP_FAILED",
            n_results=5,
        )

        # Should still find results via error_code fallback
        assert len(results) > 0

    def test_retrieve_error_recovery_fallback_to_action(self, client):
        """Test fallback to action only when error_code fails."""
        client.add_experience(
            action="robot_grasp_object",
            outcome="failure",
            error_code="DIFFERENT_ERROR",
        )

        results = client.retrieve_error_recovery_experiences(
            action="robot_grasp_object",
            error_code="GRASP_FAILED",  # Different error
            n_results=5,
        )

        # Should find results via action fallback
        assert len(results) > 0
        assert results[0]["metadata"]["action"] == "robot_grasp_object"

    def test_retrieve_error_recovery_fallback_to_semantic(self, client):
        """Test fallback to pure semantic search."""
        client.add_experience(
            action="robot_move_to",
            outcome="failure",
            error_code="MOVE_FAILED",
        )

        results = client.retrieve_error_recovery_experiences(
            action="robot_grasp_object",
            error_code="GRASP_FAILED",
            n_results=5,
        )

        # Should still return some results via semantic search
        assert isinstance(results, list)

    def test_retrieve_error_recovery_with_state_summary(self, client):
        """Test retrieval with state summary."""
        client.add_experience(
            action="robot_grasp_object",
            outcome="failure",
            error_code="GRASP_FAILED",
            state_summary={"robot_location": "table"},
        )

        results = client.retrieve_error_recovery_experiences(
            action="robot_grasp_object",
            error_code="GRASP_FAILED",
            state_summary={"robot_location": "table"},
            n_results=5,
        )

        assert len(results) > 0


# ----------------------------------------------------------------------
# TestRetrieveSuccessExperiences
# ----------------------------------------------------------------------


class TestRetrieveSuccessExperiences:
    def test_retrieve_success_experiences(self, client, sample_experience):
        """Test retrieving success experiences."""
        client.add_experience(**sample_experience)

        results = client.retrieve_success_experiences(
            action="robot_grasp_object",
            n_results=5,
        )

        assert len(results) > 0
        assert results[0]["metadata"]["outcome"] == "success"

    def test_retrieve_success_excludes_failures(self, client):
        """Test that success retrieval excludes failures."""
        client.add_experience(
            action="robot_grasp_object",
            outcome="success",
        )
        client.add_experience(
            action="robot_grasp_object",
            outcome="failure",
            error_code="GRASP_FAILED",
        )

        results = client.retrieve_success_experiences(
            action="robot_grasp_object",
            n_results=5,
        )

        assert len(results) == 1
        assert results[0]["metadata"]["outcome"] == "success"

    def test_retrieve_success_with_state_summary(self, client):
        """Test success retrieval with state summary."""
        client.add_experience(
            action="robot_grasp_object",
            outcome="success",
            state_summary={"robot_location": "table"},
        )

        results = client.retrieve_success_experiences(
            action="robot_grasp_object",
            state_summary={"robot_location": "table"},
            n_results=5,
        )

        assert len(results) > 0


# ----------------------------------------------------------------------
# TestHelpers
# ----------------------------------------------------------------------


class TestHelpers:
    def test_normalize_state_summary_none(self):
        """Test normalizing None state summary."""
        result = ChromaMemoryClient._normalize_state_summary(None)
        assert result == ""

    def test_normalize_state_summary_string(self):
        """Test normalizing string state summary."""
        result = ChromaMemoryClient._normalize_state_summary("test state")
        assert result == "test state"

    def test_normalize_state_summary_dict(self):
        """Test normalizing dict state summary."""
        result = ChromaMemoryClient._normalize_state_summary(
            {"key": "value", "number": 42}
        )
        assert "key" in result
        assert "value" in result

    def test_normalize_state_summary_truncates(self):
        """Test that long state summaries are truncated."""
        long_text = "x" * (MAX_METADATA_TEXT_CHARS + 1000)
        result = ChromaMemoryClient._normalize_state_summary(long_text)
        assert len(result) <= MAX_METADATA_TEXT_CHARS

    def test_sanitize_metadata_handles_none(self):
        """Test that None values are converted to 'unknown'."""
        result = ChromaMemoryClient._sanitize_metadata({"key": None})
        assert result["key"] == "unknown"

    def test_sanitize_metadata_preserves_scalars(self):
        """Test that scalar values are preserved."""
        metadata = {
            "string": "value",
            "int": 42,
            "float": 3.14,
            "bool": True,
        }
        result = ChromaMemoryClient._sanitize_metadata(metadata)

        assert result["string"] == "value"
        assert result["int"] == 42
        assert result["float"] == 3.14
        assert result["bool"] is True

    def test_sanitize_metadata_serializes_complex(self):
        """Test that complex values are JSON serialized."""
        metadata = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }
        result = ChromaMemoryClient._sanitize_metadata(metadata)

        assert isinstance(result["list"], str)
        assert isinstance(result["dict"], str)

    def test_build_where_empty(self):
        """Test building where clause from empty list."""
        result = ChromaMemoryClient._build_where([])
        assert result is None

    def test_build_where_single_clause(self):
        """Test building where clause from single clause."""
        result = ChromaMemoryClient._build_where([{"action": "test"}])
        assert result == {"action": "test"}

    def test_build_where_multiple_clauses(self):
        """Test building where clause from multiple clauses."""
        clauses = [
            {"action": "test"},
            {"outcome": "success"},
        ]
        result = ChromaMemoryClient._build_where(clauses)

        assert "$and" in result
        assert len(result["$and"]) == 2

    def test_build_where_filters_empty_clauses(self):
        """Test that empty clauses are filtered out."""
        clauses = [
            {"action": "test"},
            {},  # Empty clause
            {"outcome": "success"},
        ]
        result = ChromaMemoryClient._build_where(clauses)

        # Should only have 2 clauses
        assert "$and" in result
        assert len(result["$and"]) == 2

    def test_build_experience_document_minimal(self):
        """Test building document with minimal fields."""
        doc = ChromaMemoryClient.build_experience_document(
            action="test_action",
            outcome="success",
        )

        assert "Action: test_action" in doc
        assert "Outcome: success" in doc

    def test_build_experience_document_full(self):
        """Test building document with all fields."""
        doc = ChromaMemoryClient.build_experience_document(
            action="test_action",
            outcome="failure",
            state_summary="test state",
            error_code="TEST_ERROR",
            error_message="Test error message",
            resolution="Test resolution",
            task_name="test_task",
        )

        assert "Action: test_action" in doc
        assert "Outcome: failure" in doc
        assert "Task: test_task" in doc
        assert "State: test state" in doc
        assert "Error code: TEST_ERROR" in doc
        assert "Error message: Test error message" in doc
        assert "Resolution: Test resolution" in doc

    def test_build_experience_document_skips_none_error_code(self):
        """Test that NONE error code is skipped."""
        doc = ChromaMemoryClient.build_experience_document(
            action="test_action",
            outcome="success",
            error_code="NONE",
        )

        assert "Error code:" not in doc

    def test_build_experience_document_truncates(self):
        """Test that long documents are truncated."""
        long_text = "x" * (MAX_DOCUMENT_CHARS + 1000)
        doc = ChromaMemoryClient.build_experience_document(
            action="test_action",
            outcome="success",
            state_summary=long_text,
        )

        assert len(doc) <= MAX_DOCUMENT_CHARS


# ----------------------------------------------------------------------
# TestCollectionManagement
# ----------------------------------------------------------------------


class TestCollectionManagement:
    def test_count_returns_correct_number(self, client):
        """Test that count() returns correct number."""
        assert client.count() == 0

        client.add_experience(
            action="test1",
            outcome="success",
        )
        assert client.count() == 1

        client.add_experience(
            action="test2",
            outcome="success",
        )
        assert client.count() == 2

    def test_delete_collection_clears_data(self, client):
        """Test that delete_collection clears all data."""
        client.add_experience(
            action="test",
            outcome="success",
        )
        assert client.count() == 1

        client.delete_collection()
        assert client.count() == 0

    def test_delete_collection_nonexistent(self):
        """Test deleting a non-existent collection doesn't raise."""
        client = ChromaMemoryClient(
            mode="ephemeral",
            collection_name="nonexistent_test",
        )
        # Should not raise
        client.delete_collection()


# ----------------------------------------------------------------------
# TestIntegration
# ----------------------------------------------------------------------


class TestIntegration:
    def test_full_workflow(self, client):
        """Test complete workflow: add, query, retrieve."""
        # Add experiences
        client.add_experience(
            action="robot_grasp_object",
            outcome="success",
            state_summary={"robot_location": "table"},
            task_name="move_red_cup",
        )

        client.add_experience(
            action="robot_grasp_object",
            outcome="failure",
            error_code="GRASP_FAILED",
            resolution="Reposition and retry",
            task_name="move_red_cup",
        )

        client.add_experience(
            action="robot_move_to",
            outcome="success",
            task_name="move_red_cup",
        )

        assert client.count() == 3

        # Query all
        all_results = client.query("robot", n_results=10)
        assert len(all_results) == 3

        # Query specific
        grasp_results = client.retrieve_relevant_experiences(
            "grasp object",
            n_results=5,
        )
        assert len(grasp_results) > 0

        # Retrieve success
        success_results = client.retrieve_success_experiences(
            "robot_grasp_object",
            n_results=5,
        )
        assert len(success_results) == 1
        assert success_results[0]["metadata"]["outcome"] == "success"

        # Retrieve error recovery
        error_results = client.retrieve_error_recovery_experiences(
            action="robot_grasp_object",
            error_code="GRASP_FAILED",
            n_results=5,
        )
        assert len(error_results) > 0

    def test_metadata_persistence(self, client):
        """Test that metadata is correctly persisted and retrieved."""
        client.add_experience(
            action="test_action",
            outcome="success",
            error_code="NONE",
            task_id="task-123",
            task_name="test_task",
            step_index=5,
            state_summary={"key": "value"},
            metadata={"custom_field": "custom_value"},
        )

        results = client.query("test", n_results=1)
        metadata = results[0]["metadata"]

        assert metadata["action"] == "test_action"
        assert metadata["outcome"] == "success"
        assert metadata["error_code"] == "NONE"
        assert metadata["task_id"] == "task-123"
        assert metadata["task_name"] == "test_task"
        assert metadata["step_index"] == 5
        assert metadata["custom_field"] == "custom_value"
        assert "created_at" in metadata
