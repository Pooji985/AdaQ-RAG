"""Exceptions for the LLM provider layer."""


class LLMError(Exception):
    """Base exception for LLM provider errors."""


class LLMConfigurationError(LLMError):
    """Raised when LLM configuration or credentials are missing or invalid."""


class LLMGenerationError(LLMError):
    """Raised when LLM inference generation fails."""
