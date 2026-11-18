"""Data models for the RAG baseline pipeline and API."""

from typing import Any
from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    """Reference citation for a retrieved documentation chunk used in generation."""

    source_index: int = Field(description="1-based citation index corresponding to prompt context")
    chunk_id: str = Field(description="Unique chunk identifier")
    doc_id: str = Field(description="Source document identifier")
    doc_title: str = Field(description="Title of parent documentation page")
    section_title: str = Field(description="Title of documentation section")
    section_level: int = Field(default=1, description="Depth level of section heading")
    source_url: str = Field(default="", description="Original documentation URL")
    score: float = Field(description="Retrieval similarity score")
    content_snippet: str = Field(default="", description="Excerpt of chunk content for quick reference")


class RoutingInfo(BaseModel):
    """Adaptive routing and query classification metadata."""

    predicted_intent: str | None = Field(default=None, description="Classified intent (e.g. FACT_LOOKUP, COMPARISON)")
    predicted_complexity: str | None = Field(default=None, description="Classified complexity (e.g. SIMPLE, MEDIUM, COMPLEX)")
    predicted_strategy: str | None = Field(default=None, description="Selected strategy (e.g. DENSE, HYBRID, MULTI_STEP)")
    routing_reason: str | None = Field(default=None, description="Reasoning behind routing strategy decision")
    candidate_top_k: int | None = Field(default=None, description="Candidate retrieval count")
    final_top_k: int | None = Field(default=None, description="Final context chunk count")
    sub_queries: list[str] = Field(default_factory=list, description="Decomposed sub-queries if multi-step")
    was_escalated: bool = Field(default=False, description="Whether retrieval was escalated during sufficiency check")


class RAGResponse(BaseModel):
    """Grounded answer and provenance metadata returned by the RAG pipeline."""

    query: str = Field(description="Original user question")
    answer: str = Field(description="LLM-generated answer grounded in retrieved context")
    system: str = Field(default="basic_rag", description="Selected RAG system name")
    retrieval_method: str = Field(default="dense", description="Retrieval strategy name ('dense' for baseline)")
    top_k: int = Field(description="Number of context chunks requested")
    sources: list[SourceReference] = Field(default_factory=list, description="Ordered source citations")
    retrieved_chunk_ids: list[str] = Field(default_factory=list, description="IDs of all retrieved chunks")
    retrieval_scores: list[float] = Field(default_factory=list, description="Similarity scores of retrieved chunks")
    routing: RoutingInfo | None = Field(default=None, description="Routing analysis for adaptive pipelines")
    latency_ms: float = Field(default=0.0, description="Total pipeline processing latency in milliseconds")
    retrieval_calls: int | None = Field(default=None, description="Number of retrieval operations executed")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Evaluation hooks and model metadata")
