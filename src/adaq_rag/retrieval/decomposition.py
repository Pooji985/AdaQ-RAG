"""Deterministic query decomposition module for complex multi-concept queries."""

import re
from typing import Any
from pydantic import BaseModel, Field

from adaq_rag.core.config import get_settings
from adaq_rag.query_analysis.analyzer import DOMAIN_ENTITIES
from adaq_rag.query_analysis.models import (
    QueryAnalysisResult,
    QueryComplexity,
    QueryIntent,
)


class SubQuery(BaseModel):
    """Structured representation of an individual retrieval sub-query."""

    sub_query_id: str = Field(description="Unique sub-query identifier, e.g. 'sub_01'")
    query_text: str = Field(description="Focused text query to execute against retrieval indexes")
    target_concept: str = Field(description="Specific domain concept or relationship targeted")
    sub_query_type: str = Field(
        default="individual_concept",
        description="Type of sub-query: 'individual_concept', 'interaction', 'troubleshooting_stage', or 'original'",
    )
    rationale: str = Field(default="", description="Explainable rationale for generating this sub-query")


class DecomposedQuery(BaseModel):
    """Structured result of query decomposition."""

    original_query: str = Field(description="Original user query text")
    was_decomposed: bool = Field(description="True if query was decomposed into multiple sub-queries")
    sub_queries: list[SubQuery] = Field(description="Ordered, bounded list of sub-queries")
    decomposition_reason: str = Field(description="Explainable rationale for the decomposition decision")
    detected_concepts: list[str] = Field(
        default_factory=list, description="Domain entities and concepts extracted from query"
    )


class QueryDecomposer:
    """Deterministic rule-based query decomposer.

    Transforms genuinely complex, multi-concept queries into a bounded set of focused retrieval
    sub-queries targeting individual concepts and their interactions. Avoids decomposing simple queries.
    """

    def __init__(self, max_sub_queries: int | None = None) -> None:
        """Initialize QueryDecomposer.

        Args:
            max_sub_queries: Maximum number of sub-queries allowed (defaults to settings.complex_max_sub_queries).
        """
        settings = get_settings()
        self.max_sub_queries = (
            max_sub_queries if max_sub_queries is not None else settings.complex_max_sub_queries
        )
        if self.max_sub_queries <= 0:
            raise ValueError(f"max_sub_queries must be positive, got {self.max_sub_queries}")

    def decompose(
        self,
        query: str,
        analysis: QueryAnalysisResult | None = None,
    ) -> DecomposedQuery:
        """Decompose a query into focused sub-queries if complexity warrants it.

        Args:
            query: The user query string.
            analysis: Optional Phase 6 QueryAnalysisResult providing intent and complexity signals.

        Returns:
            DecomposedQuery: Structured result with bounded sub-queries.

        Raises:
            ValueError: If query is empty or not a string.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        cleaned_query = query.strip()
        detected_concepts = self._extract_concepts(cleaned_query)

        # Check whether decomposition is warranted
        should_decompose = self._should_decompose(cleaned_query, detected_concepts, analysis)

        if not should_decompose:
            # Single-step: preserve original query as the sole sub-query
            target = detected_concepts[0] if detected_concepts else "general"
            return DecomposedQuery(
                original_query=cleaned_query,
                was_decomposed=False,
                sub_queries=[
                    SubQuery(
                        sub_query_id="sub_01",
                        query_text=cleaned_query,
                        target_concept=target,
                        sub_query_type="original",
                        rationale="Query is single-concept or non-complex; decomposition not required.",
                    )
                ],
                decomposition_reason="Query is focused on a single concept; executed without decomposition.",
                detected_concepts=detected_concepts,
            )

        # Generate bounded focused sub-queries for each distinct concept
        sub_queries: list[SubQuery] = []
        intent = analysis.intent if analysis is not None else QueryIntent.MULTI_CONCEPT

        # 1. Individual concept facets (up to max_sub_queries - 1 to leave room for interaction facet)
        num_individual = min(len(detected_concepts), max(1, self.max_sub_queries - 1))
        for idx, concept in enumerate(detected_concepts[:num_individual], start=1):
            sub_id = f"sub_{idx:02d}"
            query_text = self._build_concept_query(concept, intent)
            sub_queries.append(
                SubQuery(
                    sub_query_id=sub_id,
                    query_text=query_text,
                    target_concept=concept,
                    sub_query_type="individual_concept",
                    rationale=f"Isolates documentation evidence specific to {concept}.",
                )
            )

        # 2. Joint / Interaction facet if multiple concepts exist and budget allows
        if len(detected_concepts) >= 2 and len(sub_queries) < self.max_sub_queries:
            sub_id = f"sub_{len(sub_queries) + 1:02d}"
            joint_text = self._build_interaction_query(detected_concepts[:3], intent, cleaned_query)
            sub_queries.append(
                SubQuery(
                    sub_query_id=sub_id,
                    query_text=joint_text,
                    target_concept=" + ".join(detected_concepts[:3]),
                    sub_query_type="interaction",
                    rationale="Synthesizes cross-cutting interactions and connections between concepts.",
                )
            )

        return DecomposedQuery(
            original_query=cleaned_query,
            was_decomposed=True,
            sub_queries=sub_queries[: self.max_sub_queries],
            decomposition_reason=(
                f"Complex query with {len(detected_concepts)} detected concepts decomposed into "
                f"{len(sub_queries)} bounded sub-queries covering individual facets and interaction."
            ),
            detected_concepts=detected_concepts,
        )

    def _extract_concepts(self, query: str) -> list[str]:
        """Extract unique domain concepts from query text."""
        lower_query = query.lower()
        matched: list[str] = []
        for entity in DOMAIN_ENTITIES:
            pattern = rf"\b{re.escape(entity)}\b"
            if re.search(pattern, lower_query):
                # Avoid adding redundant sub-tokens if parent already matched
                if not any(entity in existing and entity != existing for existing in matched):
                    matched.append(entity)
        return matched

    def _should_decompose(
        self,
        query: str,
        detected_concepts: list[str],
        analysis: QueryAnalysisResult | None,
    ) -> bool:
        """Determine whether decomposition should be activated."""
        if analysis is not None:
            # If explicit Phase 6 analysis is provided, respect its complexity & intent
            if analysis.complexity == QueryComplexity.COMPLEX:
                return True
            if analysis.intent == QueryIntent.MULTI_CONCEPT and len(detected_concepts) >= 2:
                return True
            # Simple or Medium single-concept queries should not decompose
            return False

        # Fallback heuristic when analysis object is not passed:
        # Require at least 2 distinct domain concepts and multi-part connectors
        if len(detected_concepts) >= 2:
            lower = query.lower()
            if any(conn in lower for conn in [" and ", " with ", " vs ", "together", "interact", "combine"]):
                return True
        return False

    def _build_concept_query(self, concept: str, intent: QueryIntent) -> str:
        """Construct a focused sub-query string for an individual concept facet."""
        if intent == QueryIntent.TROUBLESHOOTING:
            return f"Troubleshooting and common errors with {concept} in scikit-learn"
        elif intent == QueryIntent.COMPARISON:
            return f"Parameters, attributes, and behavior of {concept} in scikit-learn"
        elif intent == QueryIntent.PROCEDURE:
            return f"How to use and configure {concept} in scikit-learn"
        else:
            return f"How does {concept} work in scikit-learn?"

    def _build_interaction_query(
        self, concepts: list[str], intent: QueryIntent, original_query: str
    ) -> str:
        """Construct a synthesis sub-query for the joint relationship between concepts."""
        concepts_str = " and ".join(concepts)
        if intent == QueryIntent.TROUBLESHOOTING:
            return f"Common pipeline and integration errors combining {concepts_str}"
        elif intent == QueryIntent.COMPARISON:
            return f"Comparing and choosing between {concepts_str} in scikit-learn"
        else:
            return f"How do {concepts_str} work together and interact in scikit-learn?"
