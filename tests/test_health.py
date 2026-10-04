"""Tests for the /health endpoint."""

from fastapi.testclient import TestClient

from adaq_rag.core.config import Settings


def test_health_endpoint_status_code(client: TestClient) -> None:
    """Verify /health responds with HTTP 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_endpoint_payload(client: TestClient, test_settings: Settings) -> None:
    """Verify /health returns expected schema and metadata."""
    response = client.get("/health")
    data = response.json()

    assert data["status"] == "ok"
    assert data["app_name"] == test_settings.app_name
    assert data["version"] == test_settings.app_version
    assert data["environment"] == test_settings.environment
