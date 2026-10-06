"""Adaptive Retrieval Router for AdaQ-RAG (Phase 7)."""

from typing import Any

from adaq_rag.query_analysis.models import (
    QueryAnalysisResult,
    QueryComplexity,
    QueryIntent,
)
from adaq_rag.routing.exceptions import InvalidAnalysisResultError
from adaq_rag.routing.models import (
    RetrievalEffort,
    RetrievalPlan,
    RetrievalStrategy,
)

ROUTER_VERSION = "1.0.0"


class AdaptiveRetrievalRouter:
    """Adaptive Retrieval Router: translates Phase 6 query analysis into an actionable RetrievalPlan.

    Design Principles:
        - Decoupled from execution: Produces structured decisions without performing retrieval or model calls.
        - Primary Routing Signal: QueryComplexity determines baseline strategy, effort, and processing stages:
            * SIMPLE   -> DENSE retrieval (single-stage, no reranking, no multi-step)
            * MEDIUM   -> HYBRID retrieval (two-stage dense+sparse, with reranking, no multi-step)
            * COMPLEX  -> MULTI_STEP retrieval (deep multi-step decomposition, with reranking)
        - Intent Modulation: QueryIntent refines candidate pool size (candidate_top_k), final context
          budget (final_top_k), and tailored rationale.
        - Deterministic & Explainable: Generates consistent plans with transparent routing justifications.
    """

    def __init__(self, version: str = ROUTER_VERSION) -> None:
        self.version = version

    def route(self, analysis: Any) -> RetrievalPlan:
        """Evaluate a QueryAnalysisResult and produce a structured RetrievalPlan.

        Args:
            analysis: Validated QueryAnalysisResult from Phase 6.

        Returns:
            RetrievalPlan: Concrete plan with selected strategy, effort tier, and parameters.

        Raises:
            InvalidAnalysisResultError: If input is None, wrong type, or has invalid fields.
        """
        # Validate input analysis result
        self._validate_analysis_input(analysis)

        complexity = analysis.complexity
        intent = analysis.intent

        # Determine strategy, effort, and pipeline stages based on primary signal (Complexity)
        if complexity == QueryComplexity.SIMPLE:
            strategy = RetrievalStrategy.DENSE
            effort = RetrievalEffort.LIGHT
            use_reranking = False
            use_multi_step = False
            candidate_top_k, final_top_k, reason = self._route_simple(intent, analysis)

        elif complexity == QueryComplexity.MEDIUM:
            strategy = RetrievalStrategy.HYBRID
            effort = RetrievalEffort.MODERATE
            use_reranking = True
            use_multi_step = False
            candidate_top_k, final_top_k, reason = self._route_medium(intent, analysis)

        elif complexity == QueryComplexity.COMPLEX:
            strategy = RetrievalStrategy.MULTI_STEP
            effort = RetrievalEffort.DEEP
            use_reranking = True
            use_multi_step = True
            candidate_top_k, final_top_k, reason = self._route_complex(intent, analysis)

        else:
            raise InvalidAnalysisResultError(f"Unsupported query complexity: {complexity}")

        metadata: dict[str, Any] = {
            "intent_confidence_heuristic": analysis.intent_confidence,
            "complexity_confidence_heuristic": analysis.complexity_confidence,
            "detected_intent_signals": analysis.intent_signals,
            "detected_complexity_signals": analysis.complexity_signals,
        }

        # Note heuristic confidence flag in metadata if classification was borderline/fallback
        if analysis.complexity_confidence < 0.60 or analysis.intent_confidence <= 0.50:
            metadata["confidence_flag"] = "LOW_OR_FALLBACK_CONFIDENCE"
            metadata["confidence_note"] = (
                "Heuristic confidence is at or near fallback threshold; "
                "standard tier parameters preserved to guarantee safe retrieval coverage."
            )

        return RetrievalPlan(
            query=analysis.query,
            strategy=strategy,
            effort=effort,
            candidate_top_k=candidate_top_k,
            final_top_k=final_top_k,
            use_reranking=use_reranking,
            use_multi_step=use_multi_step,
            routing_reason=reason,
            intent=intent,
            complexity=complexity,
            intent_confidence=analysis.intent_confidence,
            complexity_confidence=analysis.complexity_confidence,
            query_analysis=analysis,
            router_version=self.version,
            metadata=metadata,
        )

    def _validate_analysis_input(self, analysis: Any) -> None:
        """Ensure input conforms to QueryAnalysisResult contract."""
        if analysis is None:
            raise InvalidAnalysisResultError("QueryAnalysisResult cannot be None.")

        if not isinstance(analysis, QueryAnalysisResult):
            raise InvalidAnalysisResultError(
                f"Expected QueryAnalysisResult instance, received {type(analysis).__name__}."
            )

        if not analysis.query or not analysis.query.strip():
            raise InvalidAnalysisResultError("Query in QueryAnalysisResult cannot be empty or whitespace-only.")

        if not isinstance(analysis.intent, QueryIntent):
            raise InvalidAnalysisResultError(f"Invalid intent type in analysis: {type(analysis.intent).__name__}.")

        if not isinstance(analysis.complexity, QueryComplexity):
            raise InvalidAnalysisResultError(
                f"Invalid complexity type in analysis: {type(analysis.complexity).__name__}."
            )

    def _route_simple(
        self, intent: QueryIntent, analysis: QueryAnalysisResult
    ) -> tuple[int, int, str]:
        """Configure parameters and explainable reason for SIMPLE tier (DENSE strategy)."""
        if intent == QueryIntent.FACT_LOOKUP:
            candidate_top_k = 5
            final_top_k = 3
            reason = (
                "Direct single-concept fact lookup routed to lightweight dense retrieval; "
                "compact candidate search (top-5) and 3 final context chunks suffice."
            )
        elif intent == QueryIntent.PROCEDURE:
            candidate_top_k = 5
            final_top_k = 4
            reason = (
                "Simple single-concept procedure routed to dense retrieval; "
                "top-5 candidate search and 4 context chunks provide concise recipe guidance."
            )
        else:
            candidate_top_k = 5
            final_top_k = 4
            reason = (
                f"Simple {intent.value} query routed to lightweight dense retrieval "
                f"without reranking; 4 context chunks selected for efficient generation."
            )

        return candidate_top_k, final_top_k, reason

    def _route_medium(
        self, intent: QueryIntent, analysis: QueryAnalysisResult
    ) -> tuple[int, int, str]:
        """Configure parameters and explainable reason for MEDIUM tier (HYBRID strategy)."""
        if intent == QueryIntent.EXPLANATION:
            candidate_top_k = 15
            final_top_k = 5
            reason = (
                "Conceptual explanation routed to hybrid retrieval (dense semantics + BM25 keyword matching) "
                "with reranking; candidate pool of 15 refined to top-5 chunks."
            )
        elif intent == QueryIntent.COMPARISON:
            candidate_top_k = 20
            final_top_k = 6
            reason = (
                "Entity comparison requires broad hybrid candidate pool (top-20) and reranking (top-6) "
                "to ensure balanced evidence across both contrasted concepts."
            )
        elif intent == QueryIntent.TROUBLESHOOTING:
            candidate_top_k = 20
            final_top_k = 6
            reason = (
                "Troubleshooting query routed to broad hybrid search (top-20) and reranking (top-6) "
                "to capture both diagnostic symptoms and resolution steps."
            )
        elif intent == QueryIntent.PROCEDURE:
            candidate_top_k = 15
            final_top_k = 5
            reason = (
                "Multi-step procedure routed to hybrid retrieval with reranking; "
                "15 candidate chunks refined to top-5 to provide complete workflow guidance."
            )
        elif intent == QueryIntent.MULTI_CONCEPT:
            candidate_top_k = 20
            final_top_k = 6
            reason = (
                "Interrelated concepts routed to expanded hybrid search with reranking (top-20 to top-6) "
                "to synthesize cross-cutting documentation sections."
            )
        else:  # FACT_LOOKUP + MEDIUM
            candidate_top_k = 12
            final_top_k = 5
            reason = (
                "Multi-attribute fact query routed to hybrid retrieval with reranking (top-12 to top-5) "
                "for comprehensive parameter coverage."
            )

        return candidate_top_k, final_top_k, reason

    def _route_complex(
        self, intent: QueryIntent, analysis: QueryAnalysisResult
    ) -> tuple[int, int, str]:
        """Configure parameters and explainable reason for COMPLEX tier (MULTI_STEP strategy)."""
        if intent == QueryIntent.MULTI_CONCEPT:
            candidate_top_k = 30
            final_top_k = 10
            reason = (
                "Complex multi-concept query routed to deep multi-step retrieval and reranking; "
                "top-30 candidate pool synthesized across distinct documentation topics into top-10 chunks."
            )
        elif intent == QueryIntent.TROUBLESHOOTING:
            candidate_top_k = 30
            final_top_k = 8
            reason = (
                "Complex multi-component troubleshooting routed to multi-step diagnostic retrieval "
                "and reranking across pipeline stages (top-30 candidates, top-8 final)."
            )
        elif intent == QueryIntent.COMPARISON:
            candidate_top_k = 25
            final_top_k = 8
            reason = (
                "Complex constrained comparison routed to multi-step retrieval and reranking "
                "to contrast multiple estimators under specific dataset conditions (top-25 candidates, top-8 final)."
            )
        elif intent == QueryIntent.EXPLANATION:
            candidate_top_k = 25
            final_top_k = 8
            reason = (
                "Complex theoretical explanation routed to deep multi-step retrieval and reranking "
                "for thorough mathematical and foundational evidence (top-25 candidates, top-8 final)."
            )
        else:
            candidate_top_k = 25
            final_top_k = 8
            reason = (
                f"Complex {intent.value} query routed to deep multi-step retrieval and reranking "
                f"to ensure sufficient multi-section coverage (top-25 candidates, top-8 final)."
            )

        return candidate_top_k, final_top_k, reason
