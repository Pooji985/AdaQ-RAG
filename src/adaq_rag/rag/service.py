"""Service-level dependency provider for the Basic RAG Pipeline."""

import logging
from fastapi import Depends
from adaq_rag.core.config import Settings, get_settings
from adaq_rag.rag.pipeline import BasicRAGPipeline

logger = logging.getLogger("adaq_rag.rag.service")

_pipeline_instance: BasicRAGPipeline | None = None


def get_rag_pipeline(settings: Settings = Depends(get_settings)) -> BasicRAGPipeline:
    """Return singleton BasicRAGPipeline instance, initializing lazily on first access.

    Args:
        settings: Optional Settings override.

    Returns:
        BasicRAGPipeline: Configured baseline pipeline.
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        logger.info("Initializing singleton BasicRAGPipeline instance")
        _pipeline_instance = BasicRAGPipeline.create(settings=settings)
    return _pipeline_instance


def reset_rag_pipeline() -> None:
    """Reset singleton instance (useful during testing)."""
    global _pipeline_instance
    _pipeline_instance = None
