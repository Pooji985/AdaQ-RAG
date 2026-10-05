"""Data models for retrieval results and index metadata."""

from typing import Any
from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """Unified retrieval result representation for dense and sparse retrievers."""

    chunk_id: str = Field(description="Unique chunk identifier")
    score: float = Field(description="Relevance or similarity score")
    retrieval_method: str = Field(description="Retrieval strategy name ('dense' or 'bm25')")
    content: str = Field(default="", description="Chunk text content")
    doc_id: str = Field(default="", description="Parent document identifier")
    doc_title: str = Field(default="", description="Parent document title")
    section_title: str = Field(default="", description="Section or subsection heading")
    section_level: int = Field(default=1, description="Heading level depth")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context metadata")
