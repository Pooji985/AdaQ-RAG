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
    dense_score: float | None = Field(default=None, description="Raw dense vector similarity score if retrieved by FAISS")
    bm25_score: float | None = Field(default=None, description="Raw BM25 score if retrieved by BM25")
    rrf_score: float | None = Field(default=None, description="Reciprocal Rank Fusion score if combined in hybrid retrieval")
    rerank_score: float | None = Field(default=None, description="Cross-encoder reranking relevance score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional context metadata")
