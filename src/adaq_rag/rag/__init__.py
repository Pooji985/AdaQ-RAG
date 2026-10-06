"""RAG package: baseline retrieval-augmented generation pipeline."""

from adaq_rag.rag.context_builder import ContextBuilder
from adaq_rag.rag.models import RAGResponse, SourceReference
from adaq_rag.rag.pipeline import BasicRAGPipeline
from adaq_rag.rag.prompts import DEFAULT_RAG_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from adaq_rag.rag.service import get_rag_pipeline, reset_rag_pipeline

__all__ = [
    "SourceReference",
    "RAGResponse",
    "ContextBuilder",
    "BasicRAGPipeline",
    "DEFAULT_RAG_SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "get_rag_pipeline",
    "reset_rag_pipeline",
]
