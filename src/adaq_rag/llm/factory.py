"""Factory for creating LLM provider instances based on configuration."""

import logging
import os

from adaq_rag.core.config import Settings, get_settings
from adaq_rag.llm.base import BaseLLMProvider
from adaq_rag.llm.exceptions import LLMConfigurationError
from adaq_rag.llm.gemini_provider import GeminiProvider
from adaq_rag.llm.mock import MockLLMProvider
from adaq_rag.llm.openai_provider import OpenAICompatibleProvider

logger = logging.getLogger("adaq_rag.llm.factory")


def get_llm_provider(settings: Settings | None = None) -> BaseLLMProvider:
    """Create and return configured LLM provider instance.

    Args:
        settings: Optional Settings instance. If None, default settings are used.

    Returns:
        BaseLLMProvider: Configured provider instance.

    Raises:
        LLMConfigurationError: If the provider is unknown or credentials are missing.
    """
    cfg = settings or get_settings()
    provider_name = cfg.llm_provider.lower().strip()

    if provider_name == "mock":
        logger.info("Initializing MockLLMProvider (model=%s)", cfg.llm_model)
        return MockLLMProvider(model=cfg.llm_model)

    if provider_name == "openai":
        return OpenAICompatibleProvider(
            api_key=cfg.llm_api_key,
            model=cfg.llm_model,
            base_url=cfg.llm_base_url,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
        )

    if provider_name == "gemini":
        return GeminiProvider(
            api_key=cfg.llm_api_key,
            model=cfg.llm_model,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
        )

    raise LLMConfigurationError(
        f"Unsupported LLM provider '{provider_name}'. Supported providers: 'openai', 'gemini', 'mock'."
    )
