"""Tests for configuration management."""

from adaq_rag.core.config import Settings


def test_default_settings() -> None:
    """Verify default configuration attributes are initialized properly."""
    settings = Settings()
    assert settings.app_name == "AdaQ-RAG API"
    assert settings.port == 8000
    assert settings.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def test_custom_settings_override() -> None:
    """Verify settings can be overridden dynamically."""
    custom = Settings(app_name="Custom-RAG", port=9000, environment="staging")
    assert custom.app_name == "Custom-RAG"
    assert custom.port == 9000
    assert custom.environment == "staging"
