"""Pytest shared test fixtures."""

import pytest
from fastapi.testclient import TestClient

from adaq_rag.api.app import create_app
from adaq_rag.core.config import Settings, get_settings


@pytest.fixture
def test_settings() -> Settings:
    """Fixture providing isolated test settings."""
    return Settings(
        app_name="AdaQ-RAG Test",
        app_version="0.1.0-test",
        environment="testing",
        debug=True,
        log_level="DEBUG",
    )


@pytest.fixture
def client(test_settings: Settings) -> TestClient:
    """Fixture providing a FastAPI TestClient configured with test settings."""
    app = create_app(settings=test_settings)
    app.dependency_overrides[get_settings] = lambda: test_settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
