"""Unit tests for Phase 8: Hybrid Retrieval and Cross-Encoder Reranking."""

from unittest.mock import MagicMock
import pytest

from adaq_rag.query_analysis.models import (
    QueryAnalysisResult,
    QueryComplexity,
    QueryIntent,
)
from adaq_rag.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from adaq_rag.retrieval.hybrid import HybridRetriever
from adaq_rag.retrieval.models import RetrievalResult
from adaq_rag.retrieval.reranker import BaseReranker, CrossEncoderReranker
from adaq_rag.retrieval.retriever import BaseRetriever
from adaq_rag.routing.models import (
    RetrievalEffort,
    RetrievalPlan,
    RetrievalStrategy,
)


@pytest.fixture
def sample_dense_results() -> list[RetrievalResult]:
    """Provide sample dense retrieval results."""
    return [
        RetrievalResult(
            chunk_id="chunk_ridge_01",
            score=0.85,
            retrieval_method="dense",
            content="Ridge regression addresses multicollinearity by adding an L2 penalty.",
            doc_id="linear_models",
            doc_title="Linear Models",
            section_title="1.1.2. Ridge",
            metadata={"source_url": "https://example.com/ridge"},
        ),
        RetrievalResult(
            chunk_id="chunk_lasso_01",
            score=0.75,
            retrieval_method="dense",
            content="Lasso regression estimates sparse coefficients using an L1 penalty.",
            doc_id="linear_models",
            doc_title="Linear Models",
            section_title="1.1.3. Lasso",
            metadata={"source_url": "https://example.com/lasso"},
        ),
        RetrievalResult(
            chunk_id="chunk_elastic_01",
            score=0.65,
            retrieval_method="dense",
            content="ElasticNet linearly combines L1 and L2 penalties.",
            doc_id="linear_models",
            doc_title="Linear Models",
            section_title="1.1.4. ElasticNet",
            metadata={"source_url": "https://example.com/elastic"},
        ),
    ]


@pytest.fixture
def sample_bm25_results() -> list[RetrievalResult]:
    """Provide sample BM25 retrieval results with partial overlap."""
    return [
        RetrievalResult(
            chunk_id="chunk_lasso_01",  # Overlap with dense (rank 1 here, rank 2 in dense)
            score=14.2,
            retrieval_method="bm25",
            content="Lasso regression estimates sparse coefficients using an L1 penalty.",
            doc_id="linear_models",
            doc_title="Linear Models",
            section_title="1.1.3. Lasso",
            metadata={"source_url": "https://example.com/lasso"},
        ),
        RetrievalResult(
            chunk_id="chunk_sgd_01",  # Distinct to BM25
            score=11.5,
            retrieval_method="bm25",
            content="SGDClassifier implements regularized linear models with SGD learning.",
            doc_id="linear_models",
            doc_title="Linear Models",
            section_title="1.1.5. SGDClassifier",
            metadata={"source_url": "https://example.com/sgd"},
        ),
        RetrievalResult(
            chunk_id="chunk_ridge_01",  # Overlap with dense (rank 3 here, rank 1 in dense)
            score=9.8,
            retrieval_method="bm25",
            content="Ridge regression addresses multicollinearity by adding an L2 penalty.",
            doc_id="linear_models",
            doc_title="Linear Models",
            section_title="1.1.2. Ridge",
            metadata={"source_url": "https://example.com/ridge"},
        ),
    ]


def test_reciprocal_rank_fusion_combines_and_deduplicates(
    sample_dense_results: list[RetrievalResult],
    sample_bm25_results: list[RetrievalResult],
) -> None:
    """Verify RRF merges candidates, eliminates duplicate chunk IDs, and computes valid RRF scores."""
    fused = reciprocal_rank_fusion(
        dense_results=sample_dense_results,
        bm25_results=sample_bm25_results,
        rrf_k=60,
    )

    # Total unique chunks across dense (3) and bm25 (3) with 2 overlapping: 4 unique chunks
    assert len(fused) == 4
    chunk_ids = [c.chunk_id for c in fused]
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found in fused pool"

    # Both chunk_ridge_01 and chunk_lasso_01 appeared in both retrievers
    # chunk_ridge_01: dense rank 1, bm25 rank 3 -> 1/(60+1) + 1/(60+3) = 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
    # chunk_lasso_01: dense rank 2, bm25 rank 1 -> 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    # chunk_lasso_01 has higher combined RRF score than chunk_ridge_01!
    top_chunk = fused[0]
    assert top_chunk.chunk_id == "chunk_lasso_01"
    assert pytest.approx(top_chunk.score, rel=1e-3) == (1 / 62 + 1 / 61)
    assert top_chunk.retrieval_method == "hybrid_rrf"
    assert top_chunk.dense_score == 0.75
    assert top_chunk.bm25_score == 14.2
    assert top_chunk.rrf_score is not None

    second_chunk = fused[1]
    assert second_chunk.chunk_id == "chunk_ridge_01"
    assert pytest.approx(second_chunk.score, rel=1e-3) == (1 / 61 + 1 / 63)

    # Chunks appearing in only one retriever
    single_chunks = [c for c in fused if c.chunk_id in ("chunk_elastic_01", "chunk_sgd_01")]
    for sc in single_chunks:
        assert sc.score < second_chunk.score


def test_reciprocal_rank_fusion_determinism_and_top_k(
    sample_dense_results: list[RetrievalResult],
    sample_bm25_results: list[RetrievalResult],
) -> None:
    """Verify repeated RRF executions produce identical order and top_k truncation works."""
    run1 = reciprocal_rank_fusion(sample_dense_results, sample_bm25_results, rrf_k=60, top_k=2)
    run2 = reciprocal_rank_fusion(sample_dense_results, sample_bm25_results, rrf_k=60, top_k=2)

    assert len(run1) == 2
    assert [c.chunk_id for c in run1] == [c.chunk_id for c in run2]
    assert [c.score for c in run1] == [c.score for c in run2]


def test_reciprocal_rank_fusion_invalid_args(
    sample_dense_results: list[RetrievalResult],
) -> None:
    """Verify validation on invalid RRF parameters."""
    with pytest.raises(ValueError):
        reciprocal_rank_fusion(sample_dense_results, [], rrf_k=0)

    with pytest.raises(ValueError):
        reciprocal_rank_fusion(sample_dense_results, [], top_k=-1)


def test_cross_encoder_reranker_scoring_and_ordering(
    sample_dense_results: list[RetrievalResult],
) -> None:
    """Verify CrossEncoderReranker sorts candidates by predicted relevance scores and preserves metadata."""
    # Mock underlying sentence-transformers CrossEncoder to test reranker in isolation
    mock_model = MagicMock()
    # Return scores: invert the initial order so chunk_elastic_01 is ranked highest
    mock_model.predict.return_value = [0.10, 0.45, 0.95]

    reranker = CrossEncoderReranker(model=mock_model)
    query = "What is ElasticNet regularization?"
    reranked = reranker.rerank(query, sample_dense_results, top_k=2)

    assert len(reranked) == 2
    # chunk_elastic_01 should now be first (score 0.95)
    assert reranked[0].chunk_id == "chunk_elastic_01"
    assert reranked[0].score == 0.95
    assert reranked[0].rerank_score == 0.95
    assert reranked[0].retrieval_method == "hybrid_reranked"
    assert reranked[0].doc_title == "Linear Models"
    assert "source_url" in reranked[0].metadata

    # chunk_lasso_01 should be second (score 0.45)
    assert reranked[1].chunk_id == "chunk_lasso_01"
    assert reranked[1].score == 0.45


def test_cross_encoder_reranker_empty_and_invalid(
    sample_dense_results: list[RetrievalResult],
) -> None:
    """Verify reranker handles empty inputs gracefully."""
    mock_model = MagicMock()
    reranker = CrossEncoderReranker(model=mock_model)

    assert reranker.rerank("", sample_dense_results) == []
    assert reranker.rerank("   ", sample_dense_results) == []
    assert reranker.rerank("valid query", []) == []

    with pytest.raises(ValueError):
        reranker.rerank("valid query", sample_dense_results, top_k=0)


def test_hybrid_retriever_orchestration_flow(
    sample_dense_results: list[RetrievalResult],
    sample_bm25_results: list[RetrievalResult],
) -> None:
    """Verify HybridRetriever queries dense and BM25 retrievers, fuses them via RRF, and reranks."""
    mock_dense = MagicMock(spec=BaseRetriever)
    mock_dense.retrieve.return_value = sample_dense_results

    mock_bm25 = MagicMock(spec=BaseRetriever)
    mock_bm25.retrieve.return_value = sample_bm25_results

    mock_reranker = MagicMock(spec=BaseReranker)
    # Reranker returns top 2
    mock_reranker.rerank.side_effect = lambda q, candidates, top_k: candidates[:top_k]

    hybrid = HybridRetriever(
        dense_retriever=mock_dense,
        bm25_retriever=mock_bm25,
        reranker=mock_reranker,
        rrf_k=60,
    )

    query = "Lasso vs Ridge regression"
    results = hybrid.retrieve(
        query=query,
        candidate_top_k=3,
        final_top_k=2,
        use_reranking=True,
    )

    assert len(results) == 2
    mock_dense.retrieve.assert_called_once_with(query, top_k=3)
    mock_bm25.retrieve.assert_called_once_with(query, top_k=3)
    mock_reranker.rerank.assert_called_once()


def test_hybrid_retriever_without_reranking(
    sample_dense_results: list[RetrievalResult],
    sample_bm25_results: list[RetrievalResult],
) -> None:
    """Verify HybridRetriever returns fused RRF results when use_reranking=False."""
    mock_dense = MagicMock(spec=BaseRetriever)
    mock_dense.retrieve.return_value = sample_dense_results

    mock_bm25 = MagicMock(spec=BaseRetriever)
    mock_bm25.retrieve.return_value = sample_bm25_results

    hybrid = HybridRetriever(
        dense_retriever=mock_dense,
        bm25_retriever=mock_bm25,
        reranker=None,
    )

    results = hybrid.retrieve("Ridge regression", candidate_top_k=3, final_top_k=2, use_reranking=False)

    assert len(results) == 2
    assert results[0].retrieval_method == "hybrid_rrf"


def test_hybrid_retriever_empty_input() -> None:
    """Verify HybridRetriever returns empty list on empty query."""
    mock_dense = MagicMock()
    mock_bm25 = MagicMock()
    hybrid = HybridRetriever(dense_retriever=mock_dense, bm25_retriever=mock_bm25)

    assert hybrid.retrieve("") == []
    assert hybrid.retrieve("   ") == []
    mock_dense.retrieve.assert_not_called()
    mock_bm25.retrieve.assert_not_called()


def test_hybrid_retriever_from_retrieval_plan(
    sample_dense_results: list[RetrievalResult],
    sample_bm25_results: list[RetrievalResult],
) -> None:
    """Verify HybridRetriever executes retrieval matching a Phase 7 MEDIUM RetrievalPlan."""
    mock_dense = MagicMock(spec=BaseRetriever)
    mock_dense.retrieve.return_value = sample_dense_results

    mock_bm25 = MagicMock(spec=BaseRetriever)
    mock_bm25.retrieve.return_value = sample_bm25_results

    mock_reranker = MagicMock(spec=BaseReranker)
    mock_reranker.rerank.side_effect = lambda q, candidates, top_k: candidates[:top_k]

    hybrid = HybridRetriever(
        dense_retriever=mock_dense,
        bm25_retriever=mock_bm25,
        reranker=mock_reranker,
    )

    # Phase 7 MEDIUM RetrievalPlan
    plan = RetrievalPlan(
        query="What is the difference between Ridge and Lasso?",
        strategy=RetrievalStrategy.HYBRID,
        effort=RetrievalEffort.MODERATE,
        candidate_top_k=20,
        final_top_k=6,
        use_reranking=True,
        use_multi_step=False,
        routing_reason="Entity comparison requires broad hybrid search and reranking.",
        intent=QueryIntent.COMPARISON,
        complexity=QueryComplexity.MEDIUM,
        intent_confidence=0.85,
        complexity_confidence=0.85,
        query_analysis=QueryAnalysisResult(
            query="What is the difference between Ridge and Lasso?",
            intent=QueryIntent.COMPARISON,
            intent_confidence=0.85,
            complexity=QueryComplexity.MEDIUM,
            complexity_confidence=0.85,
        ),
    )

    results = hybrid.retrieve_from_plan(plan)

    mock_dense.retrieve.assert_called_once_with(plan.query, top_k=20)
    mock_bm25.retrieve.assert_called_once_with(plan.query, top_k=20)
    mock_reranker.rerank.assert_called_once()
    assert len(results) <= 6


def test_hybrid_retriever_from_plan_invalid_strategy() -> None:
    """Verify retrieve_from_plan rejects plans with unsupported strategies."""
    mock_dense = MagicMock()
    mock_bm25 = MagicMock()
    hybrid = HybridRetriever(dense_retriever=mock_dense, bm25_retriever=mock_bm25)

    invalid_plan = RetrievalPlan(
        query="complex multi-step query",
        strategy=RetrievalStrategy.MULTI_STEP,
        effort=RetrievalEffort.DEEP,
        candidate_top_k=30,
        final_top_k=10,
        use_reranking=True,
        use_multi_step=True,
        routing_reason="Multi-step requires orchestrator",
        intent=QueryIntent.MULTI_CONCEPT,
        complexity=QueryComplexity.COMPLEX,
        intent_confidence=0.9,
        complexity_confidence=0.9,
        query_analysis=QueryAnalysisResult(
            query="complex multi-step query",
            intent=QueryIntent.MULTI_CONCEPT,
            intent_confidence=0.9,
            complexity=QueryComplexity.COMPLEX,
            complexity_confidence=0.9,
        ),
    )

    with pytest.raises(ValueError) as exc_info:
        hybrid.retrieve_from_plan(invalid_plan)
    assert "MULTI_STEP" in str(exc_info.value)
