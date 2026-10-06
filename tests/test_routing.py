"""Unit tests for Phase 7: Adaptive Retrieval Router."""

import pytest

from adaq_rag.query_analysis import QueryAnalyzer
from adaq_rag.query_analysis.models import (
    QueryAnalysisResult,
    QueryComplexity,
    QueryIntent,
)
from adaq_rag.routing import (
    AdaptiveRetrievalRouter,
    InvalidAnalysisResultError,
    RetrievalEffort,
    RetrievalPlan,
    RetrievalStrategy,
    RoutingError,
)


@pytest.fixture
def router() -> AdaptiveRetrievalRouter:
    """Provide a fresh AdaptiveRetrievalRouter instance."""
    return AdaptiveRetrievalRouter()


@pytest.fixture
def analyzer() -> QueryAnalyzer:
    """Provide a fresh QueryAnalyzer instance."""
    return QueryAnalyzer()


def test_simple_query_routes_to_dense_plan(router: AdaptiveRetrievalRouter) -> None:
    """Verify SIMPLE queries generate lightweight DENSE retrieval plans without reranking."""
    analysis = QueryAnalysisResult(
        query="What is StandardScaler?",
        intent=QueryIntent.FACT_LOOKUP,
        intent_confidence=0.95,
        complexity=QueryComplexity.SIMPLE,
        complexity_confidence=0.90,
    )

    plan = router.route(analysis)

    assert plan.strategy == RetrievalStrategy.DENSE
    assert plan.effort == RetrievalEffort.LIGHT
    assert plan.use_reranking is False
    assert plan.use_multi_step is False
    assert plan.candidate_top_k == 5
    assert plan.final_top_k == 3
    assert "dense" in plan.routing_reason.lower()
    assert plan.intent == QueryIntent.FACT_LOOKUP
    assert plan.complexity == QueryComplexity.SIMPLE


def test_medium_query_routes_to_hybrid_plan(router: AdaptiveRetrievalRouter) -> None:
    """Verify MEDIUM queries generate HYBRID plans with reranking enabled."""
    # 1. Medium EXPLANATION query
    expl_analysis = QueryAnalysisResult(
        query="Why does standardization help machine learning?",
        intent=QueryIntent.EXPLANATION,
        intent_confidence=0.95,
        complexity=QueryComplexity.MEDIUM,
        complexity_confidence=0.85,
    )
    expl_plan = router.route(expl_analysis)

    assert expl_plan.strategy == RetrievalStrategy.HYBRID
    assert expl_plan.effort == RetrievalEffort.MODERATE
    assert expl_plan.use_reranking is True
    assert expl_plan.use_multi_step is False
    assert expl_plan.candidate_top_k == 15
    assert expl_plan.final_top_k == 5
    assert "hybrid" in expl_plan.routing_reason.lower()

    # 2. Medium COMPARISON query
    comp_analysis = QueryAnalysisResult(
        query="What is the difference between Ridge and Lasso?",
        intent=QueryIntent.COMPARISON,
        intent_confidence=0.85,
        complexity=QueryComplexity.MEDIUM,
        complexity_confidence=0.85,
    )
    comp_plan = router.route(comp_analysis)

    assert comp_plan.strategy == RetrievalStrategy.HYBRID
    assert comp_plan.effort == RetrievalEffort.MODERATE
    assert comp_plan.use_reranking is True
    assert comp_plan.use_multi_step is False
    assert comp_plan.candidate_top_k == 20
    assert comp_plan.final_top_k == 6

    # 3. Medium TROUBLESHOOTING query
    trouble_analysis = QueryAnalysisResult(
        query="Why is my StandardScaler pipeline failing?",
        intent=QueryIntent.TROUBLESHOOTING,
        intent_confidence=0.85,
        complexity=QueryComplexity.MEDIUM,
        complexity_confidence=0.85,
    )
    trouble_plan = router.route(trouble_analysis)

    assert trouble_plan.strategy == RetrievalStrategy.HYBRID
    assert trouble_plan.use_reranking is True
    assert trouble_plan.use_multi_step is False
    assert trouble_plan.candidate_top_k == 20
    assert trouble_plan.final_top_k == 6


def test_complex_query_routes_to_multistep_plan(router: AdaptiveRetrievalRouter) -> None:
    """Verify COMPLEX queries generate MULTI_STEP plans with decomposition and reranking."""
    # 1. Complex MULTI_CONCEPT query
    multi_analysis = QueryAnalysisResult(
        query="How do preprocessing, pipelines, and model selection work together?",
        intent=QueryIntent.MULTI_CONCEPT,
        intent_confidence=0.95,
        complexity=QueryComplexity.COMPLEX,
        complexity_confidence=0.90,
    )
    multi_plan = router.route(multi_analysis)

    assert multi_plan.strategy == RetrievalStrategy.MULTI_STEP
    assert multi_plan.effort == RetrievalEffort.DEEP
    assert multi_plan.use_reranking is True
    assert multi_plan.use_multi_step is True
    assert multi_plan.candidate_top_k == 30
    assert multi_plan.final_top_k == 10
    assert "multi-step" in multi_plan.routing_reason.lower()

    # 2. Complex TROUBLESHOOTING query
    trouble_complex = QueryAnalysisResult(
        query="Pipeline throwing error when combining ColumnTransformer, PCA, and custom scorer across cross-validation",
        intent=QueryIntent.TROUBLESHOOTING,
        intent_confidence=0.90,
        complexity=QueryComplexity.COMPLEX,
        complexity_confidence=0.85,
    )
    trouble_plan = router.route(trouble_complex)

    assert trouble_plan.strategy == RetrievalStrategy.MULTI_STEP
    assert trouble_plan.effort == RetrievalEffort.DEEP
    assert trouble_plan.use_reranking is True
    assert trouble_plan.use_multi_step is True
    assert trouble_plan.candidate_top_k == 30
    assert trouble_plan.final_top_k == 8


def test_end_to_end_analyzer_to_router_integration(
    analyzer: QueryAnalyzer, router: AdaptiveRetrievalRouter
) -> None:
    """Verify realistic user queries flow seamlessly from QueryAnalyzer to AdaptiveRetrievalRouter."""
    test_cases = [
        ("What is StandardScaler?", RetrievalStrategy.DENSE, RetrievalEffort.LIGHT, False, False),
        ("Why does standardization help machine learning?", RetrievalStrategy.HYBRID, RetrievalEffort.MODERATE, True, False),
        ("How do I use StandardScaler in a pipeline?", RetrievalStrategy.HYBRID, RetrievalEffort.MODERATE, True, False),
        ("What is the difference between Ridge and Lasso?", RetrievalStrategy.HYBRID, RetrievalEffort.MODERATE, True, False),
        ("Why is my StandardScaler pipeline failing?", RetrievalStrategy.HYBRID, RetrievalEffort.MODERATE, True, False),
        (
            "How do preprocessing, pipelines, and model selection work together?",
            RetrievalStrategy.MULTI_STEP,
            RetrievalEffort.DEEP,
            True,
            True,
        ),
    ]

    for query, expected_strat, expected_effort, expected_rerank, expected_multi in test_cases:
        analysis = analyzer.analyze(query)
        plan = router.route(analysis)

        assert plan.query == query
        assert plan.strategy == expected_strat
        assert plan.effort == expected_effort
        assert plan.use_reranking == expected_rerank
        assert plan.use_multi_step == expected_multi
        assert len(plan.routing_reason) > 0


def test_confidence_values_preserved_and_flagged(router: AdaptiveRetrievalRouter) -> None:
    """Verify confidence values are preserved and heuristic fallback status is recorded in metadata."""
    # Standard high confidence
    high_conf = QueryAnalysisResult(
        query="what is Ridge?",
        intent=QueryIntent.FACT_LOOKUP,
        intent_confidence=0.95,
        complexity=QueryComplexity.SIMPLE,
        complexity_confidence=0.90,
    )
    plan_high = router.route(high_conf)
    assert plan_high.intent_confidence == 0.95
    assert plan_high.complexity_confidence == 0.90
    assert "confidence_flag" not in plan_high.metadata

    # Low/borderline confidence
    low_conf = QueryAnalysisResult(
        query="some ambiguous term",
        intent=QueryIntent.FACT_LOOKUP,
        intent_confidence=0.50,
        complexity=QueryComplexity.SIMPLE,
        complexity_confidence=0.55,
    )
    plan_low = router.route(low_conf)
    assert plan_low.intent_confidence == 0.50
    assert plan_low.complexity_confidence == 0.55
    assert plan_low.metadata.get("confidence_flag") == "LOW_OR_FALLBACK_CONFIDENCE"


def test_invalid_input_validation(router: AdaptiveRetrievalRouter) -> None:
    """Verify router raises InvalidAnalysisResultError on null, wrong type, or invalid query text."""
    with pytest.raises(InvalidAnalysisResultError) as exc_info:
        router.route(None)
    assert "cannot be None" in str(exc_info.value)

    with pytest.raises(InvalidAnalysisResultError):
        router.route("invalid string input")  # type: ignore[arg-type]

    with pytest.raises(InvalidAnalysisResultError):
        router.route({"intent": "FACT_LOOKUP"})  # type: ignore[arg-type]

    # Empty query in analysis result
    with pytest.raises(InvalidAnalysisResultError):
        invalid_analysis = QueryAnalysisResult(
            query="   ",
            intent=QueryIntent.FACT_LOOKUP,
            intent_confidence=0.5,
            complexity=QueryComplexity.SIMPLE,
            complexity_confidence=0.5,
        )
        router.route(invalid_analysis)

    # Inheritance check
    assert issubclass(InvalidAnalysisResultError, RoutingError)
    assert issubclass(InvalidAnalysisResultError, ValueError)


def test_deterministic_behavior(router: AdaptiveRetrievalRouter) -> None:
    """Verify repeated routing calls for identical analysis yield strictly identical plans."""
    analysis = QueryAnalysisResult(
        query="What is the difference between Ridge and Lasso?",
        intent=QueryIntent.COMPARISON,
        intent_confidence=0.85,
        complexity=QueryComplexity.MEDIUM,
        complexity_confidence=0.85,
    )

    plan1 = router.route(analysis)
    plan2 = router.route(analysis)

    assert plan1.model_dump() == plan2.model_dump()
    assert plan1.strategy == plan2.strategy
    assert plan1.candidate_top_k == plan2.candidate_top_k
    assert plan1.final_top_k == plan2.final_top_k
    assert plan1.routing_reason == plan2.routing_reason


def test_serialization_and_roundtrip(router: AdaptiveRetrievalRouter) -> None:
    """Verify RetrievalPlan serializes to dict and JSON with valid deserialization."""
    analysis = QueryAnalysisResult(
        query="how to use Pipeline with ColumnTransformer",
        intent=QueryIntent.PROCEDURE,
        intent_confidence=0.95,
        complexity=QueryComplexity.MEDIUM,
        complexity_confidence=0.85,
    )
    plan = router.route(analysis)

    # Dict serialization
    plan_dict = plan.model_dump()
    assert isinstance(plan_dict, dict)
    assert plan_dict["strategy"] == "HYBRID"
    assert plan_dict["effort"] == "MODERATE"
    assert plan_dict["use_reranking"] is True
    assert plan_dict["use_multi_step"] is False
    assert plan_dict["candidate_top_k"] == 15
    assert plan_dict["final_top_k"] == 5

    # JSON serialization and round-trip
    json_str = plan.model_dump_json()
    assert isinstance(json_str, str)
    deserialized = RetrievalPlan.model_validate_json(json_str)
    assert deserialized == plan
