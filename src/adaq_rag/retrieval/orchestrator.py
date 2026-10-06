from copy import deepcopy
import logging
from pathlib import Path
from typing import Any

from adaq_rag.core.config import get_settings
from adaq_rag.query_analysis.models import QueryAnalysisResult
from adaq_rag.retrieval.decomposition import DecomposedQuery, QueryDecomposer
from adaq_rag.retrieval.evidence import (
    ComplexRetrievalResult,
    EvidenceItem,
    EvidencePool,
    EvidenceSufficiencyChecker,
)
from adaq_rag.retrieval.hybrid import HybridRetriever
from adaq_rag.retrieval.models import RetrievalResult
from adaq_rag.retrieval.reranker import BaseReranker, CrossEncoderReranker
from adaq_rag.routing.models import RetrievalPlan, RetrievalStrategy

logger = logging.getLogger("adaq_rag.retrieval.orchestrator")


class ComplexRetrievalOrchestrator:
    """Orchestrates multi-step retrieval, evidence fusion, and sufficiency checks for complex queries.

    Workflow:
        Complex Query
            ↓
        Query Decomposition (QueryDecomposer)
            ↓
        Sub-query Multi-step Retrieval (HybridRetriever for each sub-query)
            ↓
        Evidence Combination with Provenance (EvidencePool)
            ↓
        Final Evidence Reranking against ORIGINAL Query (CrossEncoderReranker)
            ↓
        Evidence Sufficiency Evaluation (EvidenceSufficiencyChecker)
            ↓
        Bounded Escalation (if insufficient, up to max_attempts)
            ↓
        Final Evidence Package (ComplexRetrievalResult)
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: BaseReranker | None = None,
        decomposer: QueryDecomposer | None = None,
        sufficiency_checker: EvidenceSufficiencyChecker | None = None,
        default_candidate_top_k: int | None = None,
        default_final_top_k: int | None = None,
        max_retrieval_attempts: int | None = None,
    ) -> None:
        """Initialize the ComplexRetrievalOrchestrator with all necessary subcomponents.

        Args:
            hybrid_retriever: Pre-initialized HybridRetriever (Dense + BM25 + RRF).
            reranker: Cross-encoder reranker for final evidence prioritization.
            decomposer: Rule-based query decomposer for sub-query generation.
            sufficiency_checker: Evidence quality and coverage evaluator.
            default_candidate_top_k: Base candidates retrieved per sub-query.
            default_final_top_k: Final evidence chunks returned in package.
            max_retrieval_attempts: Maximum bounded escalation attempts allowed.
        """
        settings = get_settings()
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker or getattr(hybrid_retriever, "reranker", None) or CrossEncoderReranker()
        self.decomposer = decomposer or QueryDecomposer()
        self.sufficiency_checker = sufficiency_checker or EvidenceSufficiencyChecker()
        self.default_candidate_top_k = (
            default_candidate_top_k
            if default_candidate_top_k is not None
            else settings.complex_candidate_top_k
        )
        self.default_final_top_k = (
            default_final_top_k
            if default_final_top_k is not None
            else settings.complex_final_top_k
        )
        self.max_retrieval_attempts = (
            max_retrieval_attempts
            if max_retrieval_attempts is not None
            else settings.complex_max_retrieval_attempts
        )

        if self.max_retrieval_attempts <= 0:
            raise ValueError(
                f"max_retrieval_attempts must be positive, got {self.max_retrieval_attempts}"
            )

    @classmethod
    def load(
        cls,
        indexes_dir: str | Path | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        reranker: BaseReranker | None = None,
    ) -> "ComplexRetrievalOrchestrator":
        """Factory method to load the orchestrator using existing index artifacts.

        Args:
            indexes_dir: Optional path to index files directory.
            hybrid_retriever: Optional pre-loaded HybridRetriever instance.
            reranker: Optional custom reranker instance.

        Returns:
            ComplexRetrievalOrchestrator: Ready-to-use complex query orchestrator.
        """
        active_hybrid = hybrid_retriever or HybridRetriever.load(indexes_dir=indexes_dir, reranker=reranker)
        return cls(hybrid_retriever=active_hybrid, reranker=reranker)

    def retrieve_complex(
        self,
        query: str,
        analysis: QueryAnalysisResult | None = None,
        candidate_top_k: int | None = None,
        final_top_k: int | None = None,
        max_attempts: int | None = None,
    ) -> ComplexRetrievalResult:
        """Execute complex multi-step retrieval, evidence combination, and sufficiency evaluation.

        Args:
            query: User query text.
            analysis: Optional Phase 6 QueryAnalysisResult.
            candidate_top_k: Initial candidate pool depth per sub-query.
            final_top_k: Final evidence item count to return.
            max_attempts: Maximum escalation attempts for this query.

        Returns:
            ComplexRetrievalResult: Structured evidence package with full provenance and sufficiency state.

        Raises:
            ValueError: If query is empty or invalid.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        cleaned_query = query.strip()
        c_top_k = candidate_top_k or self.default_candidate_top_k
        f_top_k = final_top_k or self.default_final_top_k
        max_att = max_attempts or self.max_retrieval_attempts

        # 1. Decompose complex query into focused sub-queries
        decomposed = self.decomposer.decompose(cleaned_query, analysis=analysis)
        target_sub_ids = [sq.sub_query_id for sq in decomposed.sub_queries]

        attempt = 1
        current_candidate_k = c_top_k
        was_escalated = False

        # 2. Bounded retrieval and escalation loop
        while attempt <= max_att:
            # Step A: Retrieve evidence for each sub-query using HybridRetriever
            raw_sub_results: dict[str, list[RetrievalResult]] = {}
            for sq in decomposed.sub_queries:
                sub_res = self.hybrid_retriever.retrieve(
                    query=sq.query_text,
                    candidate_top_k=current_candidate_k,
                    final_top_k=current_candidate_k,
                    use_reranking=True,
                )
                raw_sub_results[sq.sub_query_id] = sub_res

            # Step B: Combine evidence across all sub-queries and preserve provenance
            pool = self._combine_evidence(
                original_query=cleaned_query,
                raw_results=raw_sub_results,
                sub_query_ids=target_sub_ids,
            )

            # Step C: Final Evidence Reranking against the ORIGINAL user query
            if pool.items:
                self._rerank_pool_against_original_query(cleaned_query, pool)

            # Step D: Evidence Sufficiency Evaluation
            sufficiency = self.sufficiency_checker.evaluate(
                evidence_pool=pool,
                target_sub_queries=target_sub_ids,
                attempt=attempt,
                max_attempts=max_att,
            )

            # Check if sufficient or if maximum bounded attempts reached
            if sufficiency.is_sufficient or attempt >= max_att:
                break

            # Trigger bounded escalation: broaden candidate retrieval depth
            was_escalated = True
            current_candidate_k = int(current_candidate_k * 1.75)
            attempt += 1

        # Select final top-k evidence items for generation
        final_evidence = pool.items[:f_top_k]

        return ComplexRetrievalResult(
            original_query=cleaned_query,
            decomposed_query=decomposed,
            evidence_pool=pool,
            sufficiency=sufficiency,
            retrieval_attempts=attempt,
            was_escalated=was_escalated,
            final_evidence=final_evidence,
            metadata={
                "candidate_depth_used": current_candidate_k,
                "sub_queries_count": len(decomposed.sub_queries),
                "total_unique_chunks_found": pool.unique_chunks_count,
            },
        )

    def retrieve_from_plan(self, plan: RetrievalPlan) -> ComplexRetrievalResult:
        """Execute retrieval as specified by a Phase 7 MULTI_STEP RetrievalPlan.

        Args:
            plan: RetrievalPlan specifying MULTI_STEP strategy and parameters.

        Returns:
            ComplexRetrievalResult: Structured evidence result package.

        Raises:
            ValueError: If the plan specifies an unsupported strategy.
        """
        if not isinstance(plan, RetrievalPlan):
            raise ValueError(f"Expected RetrievalPlan instance, got {type(plan).__name__}")

        if plan.strategy not in (RetrievalStrategy.MULTI_STEP, RetrievalStrategy.HYBRID):
            raise ValueError(
                f"ComplexRetrievalOrchestrator requires MULTI_STEP (or HYBRID) strategy, got '{plan.strategy.value}'"
            )

        return self.retrieve_complex(
            query=plan.query,
            analysis=plan.query_analysis,
            candidate_top_k=plan.candidate_top_k,
            final_top_k=plan.final_top_k,
        )

    def _combine_evidence(
        self,
        original_query: str,
        raw_results: dict[str, list[RetrievalResult]],
        sub_query_ids: list[str],
    ) -> EvidencePool:
        """Merge candidate results across all sub-queries, de-duplicate chunks, and preserve provenance."""
        evidence_by_chunk: dict[str, EvidenceItem] = {}
        total_candidates = 0

        for sq_id, results in raw_results.items():
            total_candidates += len(results)
            for res in results:
                chunk_id = res.chunk_id
                score = res.score

                if chunk_id not in evidence_by_chunk:
                    evidence_by_chunk[chunk_id] = EvidenceItem(
                        chunk_id=chunk_id,
                        content=res.content,
                        doc_id=res.doc_id,
                        doc_title=res.doc_title,
                        section_title=res.section_title,
                        section_level=res.section_level,
                        retrieved_by_sub_queries=[sq_id],
                        sub_query_scores={sq_id: score},
                        initial_score=score,
                        final_rerank_score=None,
                        overlap_count=1,
                        metadata=deepcopy(res.metadata),
                    )
                else:
                    item = evidence_by_chunk[chunk_id]
                    if sq_id not in item.retrieved_by_sub_queries:
                        item.retrieved_by_sub_queries.append(sq_id)
                    item.sub_query_scores[sq_id] = score
                    item.overlap_count = len(item.retrieved_by_sub_queries)
                    item.initial_score = max(item.initial_score, score)

        # Initial sort by initial_score
        items_list = list(evidence_by_chunk.values())
        items_list.sort(key=lambda it: (-it.initial_score, it.chunk_id))

        return EvidencePool(
            original_query=original_query,
            total_candidates_examined=total_candidates,
            unique_chunks_count=len(items_list),
            items=items_list,
            sub_query_ids=sub_query_ids,
        )

    def _rerank_pool_against_original_query(
        self,
        original_query: str,
        pool: EvidencePool,
    ) -> None:
        """Perform cross-encoder reranking of all candidate chunks against the original user query."""
        if not pool.items:
            return

        # Adapt EvidenceItems to RetrievalResult structures for the reranker interface
        adapted_candidates = [
            RetrievalResult(
                chunk_id=item.chunk_id,
                score=item.initial_score,
                retrieval_method="evidence_pool",
                content=item.content,
                doc_id=item.doc_id,
                doc_title=item.doc_title,
                section_title=item.section_title,
                section_level=item.section_level,
                metadata=item.metadata,
            )
            for item in pool.items
        ]

        reranked_results = self.reranker.rerank(
            query=original_query,
            candidates=adapted_candidates,
            top_k=len(adapted_candidates),
        )

        # Map final reranking scores back to the EvidenceItems in pool
        scores_by_id = {res.chunk_id: res.score for res in reranked_results}
        for item in pool.items:
            item.final_rerank_score = scores_by_id.get(item.chunk_id, item.initial_score)

        # Sort pool items descending by final cross-encoder score
        pool.items.sort(
            key=lambda it: (
                -(it.final_rerank_score if it.final_rerank_score is not None else it.initial_score),
                it.chunk_id,
            )
        )


# Class alias for nomenclature flexibility
MultiStepRetriever = ComplexRetrievalOrchestrator
