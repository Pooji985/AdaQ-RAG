from typing import Any
from unittest.mock import MagicMock
import pytest

from adaq_rag.query_analysis.models import (
    QueryAnalysisResult,
    QueryComplexity,
    QueryIntent,
)
from adaq_rag.retrieval.decomposition import DecomposedQuery, QueryDecomposer, SubQuery
from adaq_rag.retrieval.evidence import (
    ComplexRetrievalResult,
    EvidenceItem,
    EvidencePool,
    EvidenceSufficiencyChecker,
    SufficiencyResult,
)
from adaq_rag.retrieval.hybrid import HybridRetriever
from adaq_rag.retrieval.models import RetrievalResult
from adaq_rag.retrieval.orchestrator import (
    ComplexRetrievalOrchestrator,
    MultiStepRetriever,
)
from adaq_rag.retrieval.reranker import BaseReranker
from adaq_rag.routing.models import (
    RetrievalEffort,
    RetrievalPlan,
    RetrievalStrategy,
)


# -----------------------------------------------------------------------------
# 1. Query Decomposition Tests
# -----------------------------------------------------------------------------


def test_decomposition_simple_query_not_decomposed() -> None:
    """Verify single-concept or simple query is not decomposed."""
    decomposer = QueryDecomposer(max_sub_queries=4)
    query = "What is StandardScaler?"
    analysis = QueryAnalysisResult(
        query=query,
        intent=QueryIntent.FACT_LOOKUP,
        intent_confidence=0.95,
        complexity=QueryComplexity.SIMPLE,
        complexity_confidence=0.90,
    )

    result = decomposer.decompose(query, analysis=analysis)

    assert result.was_decomposed is False
    assert len(result.sub_queries) == 1
    assert result.sub_queries[0].query_text == query
    assert result.sub_queries[0].sub_query_type == "original"
    assert "single concept" in result.decomposition_reason.lower()


def test_decomposition_complex_query_produces_bounded_subqueries() -> None:
    """Verify complex multi-concept query generates bounded focused sub-queries."""
    decomposer = QueryDecomposer(max_sub_queries=4)
    query = "How do preprocessing, pipelines, and model selection work together?"
    analysis = QueryAnalysisResult(
        query=query,
        intent=QueryIntent.MULTI_CONCEPT,
        intent_confidence=0.95,
        complexity=QueryComplexity.COMPLEX,
        complexity_confidence=0.90,
    )

    result = decomposer.decompose(query, analysis=analysis)

    assert result.was_decomposed is True
    assert 2 <= len(result.sub_queries) <= 4
    # Verify individual concept facets and joint interaction facet
    types = [sq.sub_query_type for sq in result.sub_queries]
    assert "individual_concept" in types
    assert "interaction" in types

    # Verify original query is preserved
    assert result.original_query == query
    for sq in result.sub_queries:
        assert sq.sub_query_id.startswith("sub_")
        assert len(sq.query_text) > 0
        assert len(sq.target_concept) > 0


def test_decomposition_determinism() -> None:
    """Verify decomposition is strictly deterministic across repeated runs."""
    decomposer = QueryDecomposer(max_sub_queries=4)
    query = "compare StandardScaler, MinMaxScaler, and RobustScaler in pipeline cross-validation"

    res1 = decomposer.decompose(query)
    res2 = decomposer.decompose(query)

    assert res1.model_dump() == res2.model_dump()
    assert [sq.query_text for sq in res1.sub_queries] == [sq.query_text for sq in res2.sub_queries]


def test_decomposition_invalid_inputs() -> None:
    """Verify decomposer rejects empty, whitespace, and non-string inputs."""
    decomposer = QueryDecomposer()
    with pytest.raises(ValueError):
        decomposer.decompose("")
    with pytest.raises(ValueError):
        decomposer.decompose("   ")
    with pytest.raises(ValueError):
        decomposer.decompose(None)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# 2. Evidence Combination & Provenance Tests
# -----------------------------------------------------------------------------


def test_evidence_pool_combination_and_provenance() -> None:
    """Verify evidence items are de-duplicated, provenance is tracked, and scores preserved."""
    # Create mock raw results for 2 sub-queries with 1 overlapping chunk
    item_overlap = RetrievalResult(
        chunk_id="chunk_pipeline_01",
        score=0.85,
        retrieval_method="dense",
        content="Pipeline chains estimators sequentially.",
        doc_title="Pipelines",
        section_title="8.1. Pipeline",
    )
    item_sq1_only = RetrievalResult(
        chunk_id="chunk_prep_01",
        score=0.70,
        retrieval_method="dense",
        content="StandardScaler standardizes features by removing the mean.",
        doc_title="Preprocessing",
        section_title="6.3. Scaling",
    )
    item_sq2_only = RetrievalResult(
        chunk_id="chunk_model_01",
        score=0.90,
        retrieval_method="dense",
        content="GridSearchCV tunes hyperparameters across cross-validation splits.",
        doc_title="Model Selection",
        section_title="3.2. Hyperparameter tuning",
    )

    raw_results = {
        "sub_01": [item_overlap, item_sq1_only],
        "sub_02": [item_overlap, item_sq2_only],
    }

    mock_hybrid = MagicMock()
    orchestrator = ComplexRetrievalOrchestrator(hybrid_retriever=mock_hybrid)

    pool = orchestrator._combine_evidence(
        original_query="Combined query",
        raw_results=raw_results,
        sub_query_ids=["sub_01", "sub_02"],
    )

    # 4 raw candidates reduced to 3 unique de-duplicated chunks
    assert pool.total_candidates_examined == 4
    assert pool.unique_chunks_count == 3
    assert len(pool.items) == 3

    # Check overlap chunk provenance
    overlap_entry = next(it for it in pool.items if it.chunk_id == "chunk_pipeline_01")
    assert overlap_entry.overlap_count == 2
    assert set(overlap_entry.retrieved_by_sub_queries) == {"sub_01", "sub_02"}
    assert "sub_01" in overlap_entry.sub_query_scores
    assert "sub_02" in overlap_entry.sub_query_scores

    # Check non-overlap chunks
    single_entry = next(it for it in pool.items if it.chunk_id == "chunk_prep_01")
    assert single_entry.overlap_count == 1
    assert single_entry.retrieved_by_sub_queries == ["sub_01"]


# -----------------------------------------------------------------------------
# 3. Evidence Sufficiency Checker Tests
# -----------------------------------------------------------------------------


def test_evidence_sufficiency_sufficient_case() -> None:
    """Verify evidence pool meeting coverage, chunk count, and relevance threshold is sufficient."""
    checker = EvidenceSufficiencyChecker(min_top_score=0.0, min_coverage=0.50, min_chunks=2)

    pool = EvidencePool(
        original_query="Complex query",
        total_candidates_examined=4,
        unique_chunks_count=3,
        sub_query_ids=["sub_01", "sub_02"],
        items=[
            EvidenceItem(
                chunk_id="c1",
                content="content 1",
                retrieved_by_sub_queries=["sub_01"],
                final_rerank_score=2.5,
            ),
            EvidenceItem(
                chunk_id="c2",
                content="content 2",
                retrieved_by_sub_queries=["sub_02"],
                final_rerank_score=1.8,
            ),
        ],
    )

    result = checker.evaluate(pool, target_sub_queries=["sub_01", "sub_02"], attempt=1, max_attempts=2)

    assert result.is_sufficient is True
    assert result.recommendation == "PROCEED_TO_GENERATION"
    assert result.concept_coverage == 1.0
    assert 0.0 <= result.sufficiency_score <= 1.0


def test_evidence_sufficiency_insufficient_triggers_escalation() -> None:
    """Verify insufficient evidence on attempt 1 recommends ESCALATE_RETRIEVAL."""
    checker = EvidenceSufficiencyChecker(min_top_score=0.0, min_coverage=0.50, min_chunks=2)

    # Empty pool
    empty_pool = EvidencePool(
        original_query="Query",
        total_candidates_examined=0,
        unique_chunks_count=0,
        sub_query_ids=["sub_01", "sub_02"],
        items=[],
    )

    result = checker.evaluate(empty_pool, target_sub_queries=["sub_01", "sub_02"], attempt=1, max_attempts=2)

    assert result.is_sufficient is False
    assert result.recommendation == "ESCALATE_RETRIEVAL"
    assert "empty" in result.reason.lower()


def test_evidence_sufficiency_max_attempts_exhausted() -> None:
    """Verify persistent insufficiency on max attempt returns INSUFFICIENT_EVIDENCE."""
    checker = EvidenceSufficiencyChecker(min_top_score=0.0, min_coverage=0.50, min_chunks=2)

    empty_pool = EvidencePool(
        original_query="Query",
        total_candidates_examined=0,
        unique_chunks_count=0,
        sub_query_ids=["sub_01"],
        items=[],
    )

    result = checker.evaluate(empty_pool, target_sub_queries=["sub_01"], attempt=2, max_attempts=2)

    assert result.is_sufficient is False
    assert result.recommendation == "INSUFFICIENT_EVIDENCE"


# -----------------------------------------------------------------------------
# 4. Complex Retrieval Orchestrator Flow & Escalation Tests
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_hybrid_retriever() -> MagicMock:
    """Mock HybridRetriever returning simulated evidence."""
    mock = MagicMock(spec=HybridRetriever)
    mock.retrieve.return_value = [
        RetrievalResult(
            chunk_id="chunk_mock_01",
            score=0.8,
            retrieval_method="hybrid_rrf",
            content="Evidence chunk 1",
            doc_title="Doc 1",
            section_title="Sec 1",
        ),
        RetrievalResult(
            chunk_id="chunk_mock_02",
            score=0.7,
            retrieval_method="hybrid_rrf",
            content="Evidence chunk 2",
            doc_title="Doc 2",
            section_title="Sec 2",
        ),
    ]
    return mock


@pytest.fixture
def mock_reranker() -> MagicMock:
    """Mock CrossEncoderReranker returning ranked candidates with scores."""
    mock = MagicMock(spec=BaseReranker)
    # Assign scores: 3.5 to chunk 1, 2.0 to chunk 2
    def fake_rerank(query: str, candidates: list[RetrievalResult], top_k: int | None = None) -> list[RetrievalResult]:
        for idx, c in enumerate(candidates):
            c.score = 3.5 - idx * 1.5
        return candidates[:top_k] if top_k else candidates

    mock.rerank.side_effect = fake_rerank
    return mock


def test_orchestrator_sufficient_retrieval_flow(
    mock_hybrid_retriever: MagicMock,
    mock_reranker: MagicMock,
) -> None:
    """Verify orchestrator runs decomposition, sub-query retrieval, original-query reranking, and succeeds."""
    orchestrator = ComplexRetrievalOrchestrator(
        hybrid_retriever=mock_hybrid_retriever,
        reranker=mock_reranker,
        max_retrieval_attempts=2,
    )

    query = "How do preprocessing, pipelines, and model selection work together?"
    res = orchestrator.retrieve_complex(query=query, candidate_top_k=5, final_top_k=3)

    assert isinstance(res, ComplexRetrievalResult)
    assert res.original_query == query
    assert res.decomposed_query.was_decomposed is True
    assert res.retrieval_attempts == 1
    assert res.was_escalated is False
    assert res.sufficiency.is_sufficient is True
    assert len(res.final_evidence) <= 3
    # Check that final reranking populated final_rerank_score
    for item in res.final_evidence:
        assert item.final_rerank_score is not None


def test_orchestrator_bounded_escalation_flow(mock_reranker: MagicMock) -> None:
    """Verify orchestrator escalates retrieval when attempt 1 is insufficient and succeeds on attempt 2."""
    mock_hybrid = MagicMock(spec=HybridRetriever)

    # Call 1 (Attempt 1): returns 0 candidates -> insufficient
    # Call 2 (Attempt 2): returns valid candidates -> sufficient
    call_count = 0

    def fake_retrieve(*args: Any, **kwargs: Any) -> list[RetrievalResult]:
        nonlocal call_count
        call_count += 1
        if call_count <= 4:  # Sub-queries in attempt 1 return empty
            return []
        return [
            RetrievalResult(
                chunk_id="chunk_esc_01",
                score=0.9,
                retrieval_method="hybrid_rrf",
                content="Escalated evidence 1",
            ),
            RetrievalResult(
                chunk_id="chunk_esc_02",
                score=0.8,
                retrieval_method="hybrid_rrf",
                content="Escalated evidence 2",
            ),
        ]

    mock_hybrid.retrieve.side_effect = fake_retrieve

    orchestrator = ComplexRetrievalOrchestrator(
        hybrid_retriever=mock_hybrid,
        reranker=mock_reranker,
        max_retrieval_attempts=2,
    )

    query = "How do preprocessing, pipelines, and model selection work together?"
    res = orchestrator.retrieve_complex(query=query, candidate_top_k=5, final_top_k=3)

    assert res.retrieval_attempts == 2
    assert res.was_escalated is True
    assert res.sufficiency.is_sufficient is True


def test_orchestrator_stops_strictly_at_max_attempts(mock_reranker: MagicMock) -> None:
    """Verify orchestrator never exceeds max_retrieval_attempts and returns insufficient state."""
    mock_hybrid = MagicMock(spec=HybridRetriever)
    mock_hybrid.retrieve.return_value = []  # Always empty

    orchestrator = ComplexRetrievalOrchestrator(
        hybrid_retriever=mock_hybrid,
        reranker=mock_reranker,
        max_retrieval_attempts=2,
    )

    query = "How do preprocessing, pipelines, and model selection work together?"
    res = orchestrator.retrieve_complex(query=query, candidate_top_k=5, final_top_k=3)

    assert res.retrieval_attempts == 2
    assert res.was_escalated is True
    assert res.sufficiency.is_sufficient is False
    assert res.sufficiency.recommendation == "INSUFFICIENT_EVIDENCE"


def test_orchestrator_phase7_retrieval_plan_integration(
    mock_hybrid_retriever: MagicMock,
    mock_reranker: MagicMock,
) -> None:
    """Verify ComplexRetrievalOrchestrator seamlessly consumes a Phase 7 MULTI_STEP plan."""
    orchestrator = ComplexRetrievalOrchestrator(
        hybrid_retriever=mock_hybrid_retriever,
        reranker=mock_reranker,
    )

    plan = RetrievalPlan(
        query="How do preprocessing, pipelines, and model selection work together?",
        strategy=RetrievalStrategy.MULTI_STEP,
        effort=RetrievalEffort.DEEP,
        candidate_top_k=30,
        final_top_k=10,
        use_reranking=True,
        use_multi_step=True,
        routing_reason="Complex multi-concept query routed to deep multi-step retrieval",
        intent=QueryIntent.MULTI_CONCEPT,
        complexity=QueryComplexity.COMPLEX,
        intent_confidence=0.95,
        complexity_confidence=0.90,
        query_analysis=QueryAnalysisResult(
            query="How do preprocessing, pipelines, and model selection work together?",
            intent=QueryIntent.MULTI_CONCEPT,
            intent_confidence=0.95,
            complexity=QueryComplexity.COMPLEX,
            complexity_confidence=0.90,
        ),
    )

    res = orchestrator.retrieve_from_plan(plan)

    assert isinstance(res, ComplexRetrievalResult)
    assert res.original_query == plan.query


def test_orchestrator_rejects_unsupported_strategy(mock_hybrid_retriever: MagicMock) -> None:
    """Verify orchestrator rejects plans with DENSE strategy that should not be multi-step."""
    orchestrator = ComplexRetrievalOrchestrator(hybrid_retriever=mock_hybrid_retriever)

    dense_plan = RetrievalPlan(
        query="What is StandardScaler?",
        strategy=RetrievalStrategy.DENSE,
        effort=RetrievalEffort.LIGHT,
        candidate_top_k=5,
        final_top_k=3,
        use_reranking=False,
        use_multi_step=False,
        routing_reason="Dense lookup",
        intent=QueryIntent.FACT_LOOKUP,
        complexity=QueryComplexity.SIMPLE,
        intent_confidence=0.95,
        complexity_confidence=0.90,
        query_analysis=QueryAnalysisResult(
            query="What is StandardScaler?",
            intent=QueryIntent.FACT_LOOKUP,
            intent_confidence=0.95,
            complexity=QueryComplexity.SIMPLE,
            complexity_confidence=0.90,
        ),
    )

    with pytest.raises(ValueError) as exc_info:
        orchestrator.retrieve_from_plan(dense_plan)
    assert "MULTI_STEP" in str(exc_info.value)
