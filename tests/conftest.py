"""Shared pytest fixtures for spark-core (meta plugin)."""

import pytest
from pathlib import Path
from unittest.mock import Mock


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing agent calls without hitting real API."""
    def _create_response(content: str = "Test response", model: str = "claude-opus-4"):
        response = Mock()
        response.content = content
        response.model = model
        response.usage = Mock(input_tokens=10, output_tokens=20)
        return response
    return _create_response


@pytest.fixture
def sample_dataset():
    """Provide a minimal sample dataset for testing data processing pipelines."""
    return {
        "features": [
            {"name": "age", "type": "numeric", "missing": 0},
            {"name": "income", "type": "numeric", "missing": 2},
            {"name": "category", "type": "categorical", "missing": 0},
        ],
        "target": "income",
        "rows": 100,
        "columns": 3,
    }


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory for test artifacts."""
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    # Create subdirectories commonly used by agents
    (workspace_dir / "artifacts").mkdir()
    (workspace_dir / "reports").mkdir()
    (workspace_dir / "logs").mkdir()

    yield workspace_dir
