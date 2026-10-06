"""Data models and enums for Phase 7 Adaptive Retrieval Router."""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

from adaq_rag.query_analysis.models import (
    QueryAnalysisResult,
    QueryComplexity,
    QueryIntent,
)


class RetrievalStrategy(str, Enum):
    """Execution strategy for retrieval.

    Strategies:
        DENSE: Single-stage dense vector retrieval via FAISS for focused, single-concept queries.
        HYBRID: Two-stage retrieval combining dense vector (FAISS) and lexical (BM25) indexes with reranking.
        MULTI_STEP: Deep multi-step retrieval with query decomposition, synthesis, and reranking.
    """

    DENSE = "DENSE"
    HYBRID = "HYBRID"
    MULTI_STEP = "MULTI_STEP"


class RetrievalEffort(str, Enum):
    """Retrieval effort and depth tier.

    Tiers:
        LIGHT: Minimal candidate search and compact context window.
        MODERATE: Balanced candidate pool with cross-sectional coverage.
        DEEP: High candidate pool with multi-step or multi-concept synthesis.
    """

    LIGHT = "LIGHT"
    MODERATE = "MODERATE"
    DEEP = "DEEP"


class RetrievalPlan(BaseModel):
    """Structured plan specifying the retrieval strategy and execution parameters.

    Produced by the AdaptiveRetrievalRouter based on QueryAnalysisResult.
    Decoupled from execution: specifies HOW retrieval should be performed by downstream layers.
    """

    query: str = Field(description="Original user query text")
    strategy: RetrievalStrategy = Field(description="Selected retrieval strategy enum")
    effort: RetrievalEffort = Field(description="Retrieval effort tier")
    candidate_top_k: int = Field(
        description="Number of candidate chunks to retrieve from initial index search"
    )
    final_top_k: int = Field(
        description="Number of final context chunks to select after reranking/filtering"
    )
    use_reranking: bool = Field(
        default=False,
        description="Whether cross-encoder or neural reranking should be applied to candidates",
    )
    use_multi_step: bool = Field(
        default=False,
        description="Whether multi-step retrieval / query decomposition is required",
    )
    routing_reason: str = Field(
        description="Explainable human-readable rationale for the routing decision"
    )
    intent: QueryIntent = Field(description="Source query intent classified in Phase 6")
    complexity: QueryComplexity = Field(description="Source query complexity classified in Phase 6")
    intent_confidence: float = Field(
        description="Heuristic intent confidence from Phase 6 (not a calibrated probability)"
    )
    complexity_confidence: float = Field(
        description="Heuristic complexity confidence from Phase 6 (not a calibrated probability)"
    )
    query_analysis: QueryAnalysisResult = Field(
        description="Full source QueryAnalysisResult including diagnostic signals"
    )
    router_version: str = Field(
        default="1.0.0",
        description="Version identifier of the adaptive routing policy engine",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata and policy-tuning parameters for downstream execution",
    )
