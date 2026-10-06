"""Evidence models and sufficiency assessment module for complex retrieval."""

from copy import deepcopy
import math
from typing import Any
from pydantic import BaseModel, Field

from adaq_rag.core.config import get_settings
from adaq_rag.retrieval.decomposition import DecomposedQuery
from adaq_rag.retrieval.models import RetrievalResult


class EvidenceItem(BaseModel):
    """An individual piece of retrieved documentation evidence with full sub-query provenance."""

    chunk_id: str = Field(description="Unique chunk identifier")
    content: str = Field(description="Chunk text content")
    doc_id: str = Field(default="", description="Parent document identifier")
    doc_title: str = Field(default="", description="Parent document title")
    section_title: str = Field(default="", description="Section heading")
    section_level: int = Field(default=1, description="Heading level depth")
    retrieved_by_sub_queries: list[str] = Field(
        default_factory=list,
        description="IDs of all sub-queries that independently retrieved this chunk",
    )
    sub_query_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Score assigned to this chunk by each retrieving sub-query",
    )
    initial_score: float = Field(
        default=0.0,
        description="Highest initial candidate score from sub-query retrieval",
    )
    final_rerank_score: float | None = Field(
        default=None,
        description="Cross-encoder relevance score evaluated against the ORIGINAL query",
    )
    overlap_count: int = Field(
        default=1,
        description="Number of distinct sub-queries that independently retrieved this chunk",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Preserved chunk and document metadata",
    )


class EvidencePool(BaseModel):
    """Aggregated pool of de-duplicated evidence collected across all sub-queries."""

    original_query: str = Field(description="Original user query")
    total_candidates_examined: int = Field(
        description="Total raw candidate items processed across all sub-queries"
    )
    unique_chunks_count: int = Field(
        description="Number of unique de-duplicated chunks in the evidence pool"
    )
    items: list[EvidenceItem] = Field(
        default_factory=list,
        description="Ordered list of evidence items",
    )
    sub_query_ids: list[str] = Field(
        default_factory=list,
        description="All sub-query IDs that contributed to this pool",
    )

    def get_chunk_ids(self) -> list[str]:
        """Return list of all chunk IDs present in pool."""
        return [item.chunk_id for item in self.items]

    def get_sub_query_coverage(self) -> dict[str, int]:
        """Return count of evidence items retrieved by each sub-query."""
        counts = {sq_id: 0 for sq_id in self.sub_query_ids}
        for item in self.items:
            for sq_id in item.retrieved_by_sub_queries:
                counts[sq_id] = counts.get(sq_id, 0) + 1
        return counts


class SufficiencyResult(BaseModel):
    """Structured assessment of evidence sufficiency for complex query answer generation."""

    is_sufficient: bool = Field(description="True if evidence pool meets sufficiency threshold")
    sufficiency_score: float = Field(
        description="Heuristic sufficiency score (0.0 to 1.0), not a calibrated probability"
    )
    reason: str = Field(description="Explainable rationale for the sufficiency decision")
    concept_coverage: float = Field(
        description="Fraction of sub-queries represented in the evidence pool (0.0 to 1.0)"
    )
    covered_sub_queries: list[str] = Field(
        default_factory=list,
        description="Sub-query IDs that successfully retrieved supporting evidence",
    )
    uncovered_sub_queries: list[str] = Field(
        default_factory=list,
        description="Sub-query IDs lacking supporting evidence in pool",
    )
    top_relevance_score: float = Field(
        description="Highest cross-encoder relevance score in pool"
    )
    evidence_count: int = Field(description="Number of evidence chunks evaluated")
    recommendation: str = Field(
        description="Retrieval control decision: 'PROCEED_TO_GENERATION', 'ESCALATE_RETRIEVAL', or 'INSUFFICIENT_EVIDENCE'"
    )


class ComplexRetrievalResult(BaseModel):
    """Complete evidence package returned by Phase 9 Complex Retrieval Orchestrator."""

    original_query: str = Field(description="Original user query text")
    decomposed_query: DecomposedQuery = Field(description="Query decomposition plan and sub-queries")
    evidence_pool: EvidencePool = Field(description="Merged, de-duplicated, and reranked evidence pool")
    sufficiency: SufficiencyResult = Field(description="Evidence sufficiency assessment result")
    retrieval_attempts: int = Field(description="Number of retrieval iterations performed (bounded <= max)")
    was_escalated: bool = Field(description="True if bounded escalation was triggered")
    final_evidence: list[EvidenceItem] = Field(
        description="Final selected evidence items ready for downstream generation stage"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Diagnostic timing and execution metadata",
    )


class EvidenceSufficiencyChecker:
    """Evaluates whether an aggregated evidence pool has sufficient quality and coverage."""

    def __init__(
        self,
        min_top_score: float | None = None,
        min_coverage: float | None = None,
        min_chunks: int | None = None,
    ) -> None:
        """Initialize EvidenceSufficiencyChecker with configurable thresholds.

        Args:
            min_top_score: Minimum cross-encoder score for top chunk (defaults to settings.sufficiency_min_top_score).
            min_coverage: Minimum fraction of sub-queries covered (defaults to settings.sufficiency_min_coverage).
            min_chunks: Minimum number of evidence chunks required (defaults to settings.sufficiency_min_chunks).
        """
        settings = get_settings()
        self.min_top_score = (
            min_top_score if min_top_score is not None else settings.sufficiency_min_top_score
        )
        self.min_coverage = (
            min_coverage if min_coverage is not None else settings.sufficiency_min_coverage
        )
        self.min_chunks = (
            min_chunks if min_chunks is not None else settings.sufficiency_min_chunks
        )

    def evaluate(
        self,
        evidence_pool: EvidencePool,
        target_sub_queries: list[str],
        attempt: int = 1,
        max_attempts: int = 2,
    ) -> SufficiencyResult:
        """Evaluate evidence pool sufficiency against required sub-query coverage and relevance.

        Args:
            evidence_pool: Aggregated pool of de-duplicated evidence items.
            target_sub_queries: List of all sub-query IDs that should be covered.
            attempt: Current retrieval attempt number (1-indexed).
            max_attempts: Maximum bounded retrieval attempts allowed.

        Returns:
            SufficiencyResult: Structured evaluation with recommendation and reasons.
        """
        evidence_count = len(evidence_pool.items)
        num_targets = len(target_sub_queries)

        # 1. Check for empty pool
        if evidence_count == 0 or num_targets == 0:
            rec = "ESCALATE_RETRIEVAL" if attempt < max_attempts else "INSUFFICIENT_EVIDENCE"
            return SufficiencyResult(
                is_sufficient=False,
                sufficiency_score=0.0,
                reason="Evidence pool is empty; no documentation candidates retrieved.",
                concept_coverage=0.0,
                covered_sub_queries=[],
                uncovered_sub_queries=target_sub_queries,
                top_relevance_score=-999.0,
                evidence_count=0,
                recommendation=rec,
            )

        # 2. Compute sub-query coverage
        coverage_counts = evidence_pool.get_sub_query_coverage()
        covered = [sq_id for sq_id, count in coverage_counts.items() if count > 0]
        uncovered = [sq_id for sq_id, count in coverage_counts.items() if count == 0]
        coverage_fraction = len(covered) / num_targets if num_targets > 0 else 1.0

        # 3. Determine top relevance score
        top_item = evidence_pool.items[0]
        top_score = (
            top_item.final_rerank_score
            if top_item.final_rerank_score is not None
            else top_item.initial_score
        )

        # 4. Check criteria
        has_min_chunks = evidence_count >= self.min_chunks
        has_min_score = top_score >= self.min_top_score
        has_min_coverage = coverage_fraction >= self.min_coverage

        # Compute heuristic sufficiency score in [0.0, 1.0]
        # Sigmoid-normalized score contribution from top relevance
        norm_score = 1.0 / (1.0 + math.exp(-max(-5.0, min(5.0, top_score))))
        chunk_ratio = min(1.0, evidence_count / max(1, self.min_chunks * 2))
        heuristic_score = round(
            0.4 * coverage_fraction + 0.3 * chunk_ratio + 0.3 * norm_score,
            3,
        )

        is_sufficient = has_min_chunks and has_min_score and has_min_coverage

        if is_sufficient:
            recommendation = "PROCEED_TO_GENERATION"
            reason = (
                f"Evidence pool is sufficient: {evidence_count} chunks retrieved covering "
                f"{len(covered)}/{num_targets} sub-queries ({coverage_fraction:.0%}) with "
                f"top relevance score {top_score:.2f} (>= {self.min_top_score:.1f})."
            )
        else:
            reasons: list[str] = []
            if not has_min_chunks:
                reasons.append(f"insufficient chunks ({evidence_count} < {self.min_chunks})")
            if not has_min_score:
                reasons.append(f"low relevance ({top_score:.2f} < {self.min_top_score:.1f})")
            if not has_min_coverage:
                reasons.append(
                    f"insufficient concept coverage ({coverage_fraction:.0%} < {self.min_coverage:.0%}, missing: {uncovered})"
                )

            if attempt < max_attempts:
                recommendation = "ESCALATE_RETRIEVAL"
                reason = (
                    f"Evidence is insufficient on attempt {attempt}/{max_attempts}: {'; '.join(reasons)}. "
                    f"Escalating candidate depth for next attempt."
                )
            else:
                recommendation = "INSUFFICIENT_EVIDENCE"
                reason = (
                    f"Evidence remains insufficient after reaching maximum attempts ({attempt}/{max_attempts}): "
                    f"{'; '.join(reasons)}. Returning insufficient-evidence status."
                )

        return SufficiencyResult(
            is_sufficient=is_sufficient,
            sufficiency_score=heuristic_score,
            reason=reason,
            concept_coverage=coverage_fraction,
            covered_sub_queries=covered,
            uncovered_sub_queries=uncovered,
            top_relevance_score=top_score,
            evidence_count=evidence_count,
            recommendation=recommendation,
        )
