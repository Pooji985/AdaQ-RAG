"""Unit tests for Phase 6 Part 1: Query Analysis Engine."""

import pytest

from adaq_rag.query_analysis import (
    InvalidQueryError,
    QueryAnalysisError,
    QueryAnalysisResult,
    QueryAnalyzer,
    QueryComplexity,
    QueryIntent,
)


@pytest.fixture
def analyzer() -> QueryAnalyzer:
    """Provide a fresh QueryAnalyzer instance."""
    return QueryAnalyzer()


def test_input_validation_empty_and_invalid(analyzer: QueryAnalyzer) -> None:
    """Verify analyzer rejects empty, whitespace, and non-string inputs."""
    with pytest.raises(InvalidQueryError) as exc_info:
        analyzer.analyze("")
    assert "empty or whitespace" in str(exc_info.value).lower()

    with pytest.raises(InvalidQueryError):
        analyzer.analyze("   \t\n  ")

    with pytest.raises(InvalidQueryError):
        analyzer.analyze(None)  # type: ignore[arg-type]

    with pytest.raises(InvalidQueryError):
        analyzer.analyze(12345)  # type: ignore[arg-type]

    # Verify inheritance hierarchy
    assert issubclass(InvalidQueryError, QueryAnalysisError)
    assert issubclass(InvalidQueryError, ValueError)


def test_all_six_intents_classified_correctly(analyzer: QueryAnalyzer) -> None:
    """Verify representative queries for all six intent categories."""
    test_cases = [
        # FACT_LOOKUP
        ("what is StandardScaler?", QueryIntent.FACT_LOOKUP),
        ("what is the default value of ccp_alpha in DecisionTreeClassifier?", QueryIntent.FACT_LOOKUP),
        # EXPLANATION
        ("why does StandardScaler work this way?", QueryIntent.EXPLANATION),
        ("explain how gradient boosting optimizes its loss function", QueryIntent.EXPLANATION),
        # PROCEDURE
        ("how do I use StandardScaler in scikit-learn?", QueryIntent.PROCEDURE),
        ("how to configure cross-validation with StratifiedKFold", QueryIntent.PROCEDURE),
        # COMPARISON
        ("what is the difference between StandardScaler and MinMaxScaler?", QueryIntent.COMPARISON),
        ("compare Ridge vs Lasso regression", QueryIntent.COMPARISON),
        # TROUBLESHOOTING
        ("why is my StandardScaler pipeline failing?", QueryIntent.TROUBLESHOOTING),
        ("how to fix ConvergenceWarning in LogisticRegression", QueryIntent.TROUBLESHOOTING),
        # MULTI_CONCEPT
        (
            "How do cross-validation, hyperparameter tuning with GridSearchCV, and Pipeline interact in scikit-learn?",
            QueryIntent.MULTI_CONCEPT,
        ),
        (
            "Explain the relationship between bias-variance tradeoff and regularization parameters in Ridge and Lasso",
            QueryIntent.MULTI_CONCEPT,
        ),
    ]

    for query, expected_intent in test_cases:
        result = analyzer.analyze(query)
        assert result.intent == expected_intent, f"Query '{query}' expected {expected_intent}, got {result.intent}"
        assert 0.0 <= result.intent_confidence <= 1.0
        assert len(result.intent_signals) > 0


def test_all_three_complexity_levels(analyzer: QueryAnalyzer) -> None:
    """Verify all three complexity levels are represented with evidence-based scoring."""
    # SIMPLE: Focused single-concept fact lookups
    simple_result = analyzer.analyze("what is StandardScaler?")
    assert simple_result.complexity == QueryComplexity.SIMPLE
    assert 0.0 <= simple_result.complexity_confidence <= 1.0

    # MEDIUM: Standard procedures, explanations, and binary comparisons
    medium_proc = analyzer.analyze("how do I use StandardScaler in scikit-learn?")
    assert medium_proc.complexity == QueryComplexity.MEDIUM

    medium_comp = analyzer.analyze("what is the difference between StandardScaler and MinMaxScaler?")
    assert medium_comp.complexity == QueryComplexity.MEDIUM

    medium_trouble = analyzer.analyze("why is my StandardScaler pipeline failing?")
    assert medium_trouble.complexity == QueryComplexity.MEDIUM

    # COMPLEX: Multi-concept integrations and constrained multi-part queries
    complex_multi = analyzer.analyze(
        "How do cross-validation, hyperparameter tuning with GridSearchCV, and Pipeline interact in scikit-learn?"
    )
    assert complex_multi.complexity == QueryComplexity.COMPLEX

    complex_constrained = analyzer.analyze(
        "compare StandardScaler and MinMaxScaler and explain when to use each for datasets with outliers"
    )
    assert complex_constrained.complexity == QueryComplexity.COMPLEX
    assert 0.0 <= complex_constrained.complexity_confidence <= 1.0


def test_rule_priority_and_deterministic_precedence(analyzer: QueryAnalyzer) -> None:
    """Verify deterministic priority hierarchy resolves overlapping signals consistently."""
    # 1. TROUBLESHOOTING beats EXPLANATION ("why is ... failing")
    trouble_res = analyzer.analyze("why is my StandardScaler pipeline failing?")
    assert trouble_res.intent == QueryIntent.TROUBLESHOOTING
    assert any("TROUBLESHOOTING" in s for s in trouble_res.intent_signals)
    assert any("subordinated" in s for s in trouble_res.intent_signals)

    # 2. COMPARISON beats FACT_LOOKUP ("what is the difference between ...")
    comp_fact_res = analyzer.analyze("what is the difference between StandardScaler and MinMaxScaler?")
    assert comp_fact_res.intent == QueryIntent.COMPARISON

    # 3. COMPARISON beats EXPLANATION ("compare ... and explain when to use each")
    comp_explain_res = analyzer.analyze(
        "compare StandardScaler and MinMaxScaler and explain when to use each"
    )
    assert comp_explain_res.intent == QueryIntent.COMPARISON
    assert any("subordinated EXPLANATION" in s for s in comp_explain_res.intent_signals)


def test_deterministic_repeated_results(analyzer: QueryAnalyzer) -> None:
    """Verify identical queries yield strictly identical results and signals."""
    query = "what is the difference between Ridge and Lasso?"
    res1 = analyzer.analyze(query)
    res2 = analyzer.analyze(query)

    assert res1.model_dump() == res2.model_dump()
    assert res1.intent == res2.intent
    assert res1.complexity == res2.complexity
    assert res1.intent_confidence == res2.intent_confidence
    assert res1.complexity_confidence == res2.complexity_confidence
    assert res1.intent_signals == res2.intent_signals
    assert res1.complexity_signals == res2.complexity_signals


def test_serialization_and_schema_compatibility(analyzer: QueryAnalyzer) -> None:
    """Verify QueryAnalysisResult serializes cleanly to dict and JSON."""
    result = analyzer.analyze("how to use Pipeline with ColumnTransformer")

    # Dictionary serialization
    res_dict = result.model_dump()
    assert isinstance(res_dict, dict)
    assert res_dict["intent"] == "PROCEDURE"
    assert res_dict["complexity"] in ["MEDIUM", "COMPLEX"]
    assert isinstance(res_dict["intent_confidence"], float)
    assert isinstance(res_dict["complexity_confidence"], float)
    assert isinstance(res_dict["intent_signals"], list)
    assert isinstance(res_dict["complexity_signals"], list)
    assert res_dict["analyzer_version"] == "1.0.0"

    # JSON serialization and round-trip deserialization
    json_str = result.model_dump_json()
    assert isinstance(json_str, str)
    deserialized = QueryAnalysisResult.model_validate_json(json_str)
    assert deserialized == result
