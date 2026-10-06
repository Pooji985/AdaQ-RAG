"""API routes module."""

from adaq_rag.api.routes.health import router as health_router
from adaq_rag.api.routes.rag import router as rag_router

__all__ = ["health_router", "rag_router"]
