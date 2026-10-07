"""Focused test suite for AdaQ-RAG evaluation foundation (Part 1).

Tests cover:
- Recall@K
- MRR (Reciprocal Rank and Mean Reciprocal Rank)
- nDCG@K
- Routing accuracy
- Score validation (0-4 bounds)
- Overall score arithmetic mean calculation
- MockJudge and LLMJudge
- Efficiency tracking and aggregation
- Full model serialization and round-trip deserialization
"""

import math
import pytest
from pydantic import ValidationError

from adaq_rag.llm.mock import MockLLMProvider
from adaq_rag.llm.models import LLMUsage
from adaq_rag.rag.models import RAGResponse
from evaluation.rag.efficiency import (
    EfficiencyTracker,
    aggregate_efficiency_metrics,
    extract_efficiency_from_response,
)
from evaluation.rag.judge import BaseJudge, LLMJudge, MockJudge, format_evidence
from evaluation.rag.metrics import (
    compute_retrieval_metrics,
    compute_routing_accuracy,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.rag.models import (
    AnswerQualityScores,
    EfficiencyMetrics,
    QuestionEvaluationResult,
    RetrievalMetrics,
    RoutingEvaluation,
    interpret_score,
)


# =============================================================================
# 1. RETRIEVAL METRICS TESTS: Recall@K
# =============================================================================


class TestRecallAtK:
    """Tests for Recall@K metric calculation."""

    def test_recall_at_k_single_ground_truth(self) -> None:
        gt = ["chunk_a"]
        assert recall_at_k(["chunk_a", "chunk_b"], gt, k=1) == 1.0
        assert recall_at_k(["chunk_b", "chunk_a"], gt, k=1) == 0.0
        assert recall_at_k(["chunk_b", "chunk_a"], gt, k=2) == 1.0

    def test_recall_at_k_multiple_ground_truth(self) -> None:
        gt = ["chunk_a", "chunk_b"]
        retrieved = ["chunk_a", "chunk_c", "chunk_b"]

        assert recall_at_k(retrieved, gt, k=1) == 0.5
        assert recall_at_k(retrieved, gt, k=2) == 0.5
        assert recall_at_k(retrieved, gt, k=3) == 1.0

    def test_recall_at_k_no_hits(self) -> None:
        gt = ["chunk_a", "chunk_b"]
        retrieved = ["chunk_x", "chunk_y", "chunk_z"]
        assert recall_at_k(retrieved, gt, k=3) == 0.0

    def test_recall_at_k_edge_cases(self) -> None:
        # Invalid k
        assert recall_at_k(["c1"], ["c1"], k=0) == 0.0
        assert recall_at_k(["c1"], ["c1"], k=-1) == 0.0

        # Empty lists
        assert recall_at_k([], ["c1"], k=5) == 0.0
        assert recall_at_k(["c1"], [], k=5) == 0.0
        assert recall_at_k([], [], k=5) == 0.0

        # k larger than retrieved list length
        assert recall_at_k(["c1"], ["c1", "c2"], k=10) == 0.5

    def test_recall_at_k_does_not_double_count_duplicates(self) -> None:
        gt = ["chunk_a", "chunk_b"]
        # Retrieved list repeats chunk_a
        retrieved = ["chunk_a", "chunk_a", "chunk_c"]
        # In top 2, only chunk_a is hit, so recall is 1 / 2 = 0.5, not 2 / 2
        assert recall_at_k(retrieved, gt, k=2) == 0.5


# =============================================================================
# 2. RETRIEVAL METRICS TESTS: MRR & Reciprocal Rank
# =============================================================================


class TestReciprocalRankAndMRR:
    """Tests for Reciprocal Rank and Mean Reciprocal Rank."""

    def test_reciprocal_rank_ranks(self) -> None:
        gt = ["target_chunk"]

        # Rank 1 hit -> 1.0
        assert reciprocal_rank(["target_chunk", "other"], gt) == 1.0

        # Rank 2 hit -> 0.5
        assert reciprocal_rank(["other", "target_chunk"], gt) == 0.5

        # Rank 3 hit -> 1/3
        assert pytest.approx(reciprocal_rank(["o1", "o2", "target_chunk"], gt), 0.001) == 1 / 3

        # Rank 4 hit -> 0.25
        assert reciprocal_rank(["o1", "o2", "o3", "target_chunk"], gt) == 0.25

    def test_reciprocal_rank_no_hit(self) -> None:
        gt = ["target_chunk"]
        assert reciprocal_rank(["o1", "o2", "o3"], gt) == 0.0

    def test_reciprocal_rank_with_cutoff_k(self) -> None:
        gt = ["target_chunk"]
        retrieved = ["o1", "o2", "target_chunk"]
        # Hit is at rank 3
        assert reciprocal_rank(retrieved, gt, k=2) == 0.0
        assert pytest.approx(reciprocal_rank(retrieved, gt, k=3), 0.001) == 1 / 3

    def test_reciprocal_rank_edge_cases(self) -> None:
        assert reciprocal_rank([], ["c1"]) == 0.0
        assert reciprocal_rank(["c1"], []) == 0.0
        assert reciprocal_rank(["c1"], ["c1"], k=0) == 0.0
        assert reciprocal_rank(["c1"], ["c1"], k=-5) == 0.0

    def test_mean_reciprocal_rank_across_queries(self) -> None:
        queries_ret = [
            ["c1", "c2"],       # Hit at rank 1 -> RR = 1.0
            ["other", "c2"],    # Hit at rank 2 -> RR = 0.5
            ["miss1", "miss2"], # No hit -> RR = 0.0
        ]
        queries_gt = [
            ["c1"],
            ["c2"],
            ["c3"],
        ]
        # Mean RR: (1.0 + 0.5 + 0.0) / 3 = 0.5
        assert pytest.approx(mean_reciprocal_rank(queries_ret, queries_gt), 0.001) == 0.5

    def test_mean_reciprocal_rank_edge_cases(self) -> None:
        assert mean_reciprocal_rank([], []) == 0.0

        with pytest.raises(ValueError):
            mean_reciprocal_rank([["c1"]], [["c1"], ["c2"]])


# =============================================================================
# 3. RETRIEVAL METRICS TESTS: nDCG@K
# =============================================================================


class TestNDCGAtK:
    """Tests for nDCG@K calculation."""

    def test_ndcg_at_k_perfect_ranking(self) -> None:
        gt = ["c1", "c2"]
        retrieved = ["c1", "c2", "c3", "c4"]
        # Perfect ordering -> nDCG is 1.0 for any k >= 1
        assert ndcg_at_k(retrieved, gt, k=2) == 1.0
        assert ndcg_at_k(retrieved, gt, k=4) == 1.0

    def test_ndcg_at_k_suboptimal_ranking(self) -> None:
        gt = ["target"]
        retrieved = ["irrelevant", "target"]
        # Rank 1: irrelevant (0)
        # Rank 2: target (1)
        # DCG@2 = 1 / log2(2 + 1) = 1 / log2(3) ≈ 0.63092975
        # IDCG@2 = 1 / log2(1 + 1) = 1 / log2(2) = 1.0
        # nDCG@2 = 0.63092975 / 1.0 ≈ 0.6309
        expected = 1.0 / math.log2(3)
        assert pytest.approx(ndcg_at_k(retrieved, gt, k=2), 0.0001) == expected

    def test_ndcg_at_k_no_hits(self) -> None:
        gt = ["target"]
        retrieved = ["c1", "c2", "c3"]
        assert ndcg_at_k(retrieved, gt, k=3) == 0.0

    def test_ndcg_at_k_edge_cases(self) -> None:
        assert ndcg_at_k(["c1"], ["c1"], k=0) == 0.0
        assert ndcg_at_k(["c1"], ["c1"], k=-2) == 0.0
        assert ndcg_at_k([], ["c1"], k=5) == 0.0
        assert ndcg_at_k(["c1"], [], k=5) == 0.0

    def test_ndcg_at_k_duplicate_retrieved_handling(self) -> None:
        gt = ["c1"]
        retrieved = ["c1", "c1", "c2"]
        # Should NOT count c1 twice
        # DCG = 1 / log2(2) = 1.0
        # IDCG = 1.0
        assert ndcg_at_k(retrieved, gt, k=3) == 1.0

    def test_compute_retrieval_metrics_bundle(self) -> None:
        retrieved = ["c1", "c2", "c3"]
        gt = ["c2"]
        bundle = compute_retrieval_metrics(retrieved, gt, k_values=(1, 3))

        assert bundle.recall_at_k[1] == 0.0
        assert bundle.recall_at_k[3] == 1.0
        assert bundle.hit_at_k[1] is False
        assert bundle.hit_at_k[3] is True
        assert bundle.mrr == 0.5
        assert bundle.retrieved_chunk_ids == retrieved
        assert bundle.ground_truth_chunk_ids == gt


# =============================================================================
# 4. ROUTING ACCURACY TESTS
# =============================================================================


class TestRoutingAccuracy:
    """Tests for router strategy classification accuracy."""

    def test_routing_accuracy_perfect(self) -> None:
        pred = ["DENSE", "HYBRID", "MULTI_STEP"]
        exp = ["DENSE", "HYBRID", "MULTI_STEP"]
        assert compute_routing_accuracy(pred, exp) == 1.0

    def test_routing_accuracy_partial(self) -> None:
        pred = ["DENSE", "HYBRID", "DENSE", "MULTI_STEP"]
        exp = ["DENSE", "HYBRID", "MULTI_STEP", "MULTI_STEP"]
        # 3 out of 4 match -> 0.75
        assert compute_routing_accuracy(pred, exp) == 0.75

    def test_routing_accuracy_case_insensitive_and_whitespace(self) -> None:
        pred = ["dense ", "hybrid", "Multi_Step"]
        exp = ["DENSE", "HYBRID", "MULTI_STEP"]
        assert compute_routing_accuracy(pred, exp) == 1.0

    def test_routing_accuracy_edge_cases(self) -> None:
        assert compute_routing_accuracy([], []) == 0.0

        with pytest.raises(ValueError):
            compute_routing_accuracy(["DENSE"], ["DENSE", "HYBRID"])


# =============================================================================
# 5. SCORE VALIDATION & OVERALL CALCULATION TESTS
# =============================================================================


class TestScoreValidationAndCalculation:
    """Tests for AnswerQualityScores validation and overall arithmetic mean."""

    def test_valid_scores_creation(self) -> None:
        scores = AnswerQualityScores(
            correctness=3.5,
            faithfulness=4.0,
            relevance=3.0,
            reason="Clear and accurate answer.",
        )
        assert scores.correctness == 3.5
        assert scores.faithfulness == 4.0
        assert scores.relevance == 3.0
        # Expected arithmetic mean: (3.5 + 4.0 + 3.0) / 3 = 3.5
        assert scores.overall == 3.5
        assert scores.reason == "Clear and accurate answer."

    def test_arithmetic_mean_overall_auto_computed(self) -> None:
        scores = AnswerQualityScores(
            correctness=1.0,
            faithfulness=2.0,
            relevance=3.0,
        )
        # Expected arithmetic mean: (1.0 + 2.0 + 3.0) / 3 = 2.0
        assert scores.overall == 2.0

    def test_explicit_overall_preserved(self) -> None:
        scores = AnswerQualityScores(
            correctness=2.0,
            faithfulness=2.0,
            relevance=2.0,
            overall=2.5,  # Explicitly passed
        )
        assert scores.overall == 2.5

    def test_score_bounds_validation(self) -> None:
        # Negative score
        with pytest.raises(ValidationError):
            AnswerQualityScores(correctness=-0.1, faithfulness=3.0, relevance=3.0)

        # Score > 4.0
        with pytest.raises(ValidationError):
            AnswerQualityScores(correctness=4.1, faithfulness=3.0, relevance=3.0)

        with pytest.raises(ValidationError):
            AnswerQualityScores(correctness=3.0, faithfulness=5.0, relevance=3.0)

        with pytest.raises(ValidationError):
            AnswerQualityScores(correctness=3.0, faithfulness=3.0, relevance=-1.0)

    def test_score_interpretation_bands(self) -> None:
        # 0 = does not satisfy
        assert interpret_score(0.0) == "does not satisfy"

        # 0–<1 = very poor
        assert interpret_score(0.1) == "very poor"
        assert interpret_score(0.99) == "very poor"

        # 1–<2 = partially satisfies
        assert interpret_score(1.0) == "partially satisfies"
        assert interpret_score(1.5) == "partially satisfies"
        assert interpret_score(1.99) == "partially satisfies"

        # 2–<3 = mostly satisfactory
        assert interpret_score(2.0) == "mostly satisfactory"
        assert interpret_score(2.5) == "mostly satisfactory"
        assert interpret_score(2.99) == "mostly satisfactory"

        # 3–4 = strongly satisfies
        assert interpret_score(3.0) == "strongly satisfies"
        assert interpret_score(3.5) == "strongly satisfies"
        assert interpret_score(4.0) == "strongly satisfies"

        # Out of bounds
        with pytest.raises(ValueError):
            interpret_score(-0.5)
        with pytest.raises(ValueError):
            interpret_score(4.5)

    def test_score_property_interpretation(self) -> None:
        scores = AnswerQualityScores(correctness=3.5, faithfulness=3.5, relevance=3.5)
        assert scores.interpretation == "strongly satisfies"


# =============================================================================
# 6. JUDGE TESTS: MockJudge & LLMJudge
# =============================================================================


class TestJudgeInterface:
    """Tests for MockJudge and LLMJudge."""

    def test_mock_judge_default_behavior(self) -> None:
        judge = MockJudge()
        scores = judge.evaluate(
            question="What is Ridge regression?",
            reference_answer="Ridge uses L2 penalty.",
            retrieved_evidence="Ridge regression solves a model with L2 regularization.",
            generated_answer="Ridge regression applies an L2 penalty to linear coefficients.",
        )

        assert isinstance(scores, AnswerQualityScores)
        assert 0.0 <= scores.correctness <= 4.0
        assert 0.0 <= scores.faithfulness <= 4.0
        assert 0.0 <= scores.relevance <= 4.0
        assert scores.overall == scores.correctness  # all 3.5
        assert len(judge.call_history) == 1

        # Check call history does NOT contain system name
        call_record = judge.call_history[0]
        assert "system_name" not in call_record
        assert "question" in call_record
        assert "reference_answer" in call_record
        assert "retrieved_evidence" in call_record
        assert "generated_answer" in call_record

    def test_mock_judge_empty_answer(self) -> None:
        judge = MockJudge()
        scores = judge.evaluate(
            question="What is Ridge regression?",
            reference_answer="Ridge uses L2 penalty.",
            retrieved_evidence="Some context",
            generated_answer="",
        )
        assert scores.correctness == 0.0
        assert scores.overall == 0.0

    def test_mock_judge_custom_scores(self) -> None:
        custom = AnswerQualityScores(
            correctness=2.0, faithfulness=3.0, relevance=4.0, reason="Custom test score"
        )
        judge = MockJudge(default_scores=custom)
        scores = judge.evaluate("q", "ref", "ev", "gen")
        assert scores.correctness == 2.0
        assert scores.faithfulness == 3.0
        assert scores.relevance == 4.0
        assert scores.reason == "Custom test score"

    def test_mock_judge_custom_scoring_fn(self) -> None:
        def custom_fn(q: str, ref: str, ev: str | list[str], gen: str) -> AnswerQualityScores:
            return AnswerQualityScores(
                correctness=1.0, faithfulness=1.0, relevance=1.0, reason=f"Evaluated: {q}"
            )

        judge = MockJudge(scoring_fn=custom_fn)
        scores = judge.evaluate("Test Question", "ref", "ev", "gen")
        assert scores.overall == 1.0
        assert scores.reason == "Evaluated: Test Question"

    @pytest.mark.anyio
    async def test_mock_judge_async(self) -> None:
        judge = MockJudge()
        scores = await judge.evaluate_async("q", "ref", "ev", "gen answer")
        assert scores.overall > 0.0

    def test_format_evidence_helper(self) -> None:
        assert format_evidence("") == "No evidence provided."
        assert format_evidence([]) == "No evidence provided."
        assert format_evidence("Direct string evidence") == "Direct string evidence"

        snippets = ["First chunk content", "Second chunk content"]
        formatted = format_evidence(snippets)
        assert "[Evidence Snippet 1]" in formatted
        assert "First chunk content" in formatted
        assert "[Evidence Snippet 2]" in formatted

    def test_llm_judge_with_mock_provider(self) -> None:
        json_output = (
            '{"correctness": 3.8, "faithfulness": 4.0, "relevance": 3.6, '
            '"reason": "Accurate and grounded in documentation context."}'
        )
        mock_provider = MockLLMProvider(default_response=json_output)
        judge = LLMJudge(provider=mock_provider)

        scores = judge.evaluate(
            question="What is the default solver in LogisticRegression?",
            reference_answer="The default solver is lbfgs.",
            retrieved_evidence="solver : {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'}, default='lbfgs'",
            generated_answer="In scikit-learn LogisticRegression, the default solver is 'lbfgs'.",
        )

        assert scores.correctness == 3.8
        assert scores.faithfulness == 4.0
        assert scores.relevance == 3.6
        assert pytest.approx(scores.overall, 0.001) == (3.8 + 4.0 + 3.6) / 3.0
        assert "Accurate" in scores.reason

        # Ensure system prompt / prompt does NOT contain system name
        assert len(mock_provider.call_history) == 1
        prompt = mock_provider.call_history[0]["prompt"]
        assert "AdaQ-RAG" not in prompt
        assert "Baseline RAG" not in prompt


# =============================================================================
# 7. EFFICIENCY TRACKING TESTS
# =============================================================================


class TestEfficiencyUtilities:
    """Tests for EfficiencyTracker and efficiency metrics recording."""

    def test_efficiency_tracker_context_manager(self) -> None:
        with EfficiencyTracker() as tracker:
            tracker.record_retrieval(chunks_count=5)
            tracker.record_retrieval(chunks_count=3)
            tracker.record_escalation(count=1)
            tracker.record_tokens(
                LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
            )

        metrics = tracker.get_metrics()
        assert metrics.latency_ms >= 0.0
        assert metrics.retrieval_calls == 2
        assert metrics.chunks_processed == 8
        assert metrics.escalation_count == 1
        assert metrics.token_usage is not None
        assert metrics.token_usage.total_tokens == 150

    def test_efficiency_tracker_token_usage_optional_when_none(self) -> None:
        tracker = EfficiencyTracker()
        tracker.start()
        tracker.record_retrieval(chunks_count=4)
        tracker.stop()

        metrics = tracker.get_metrics()
        assert metrics.token_usage is None
        assert metrics.retrieval_calls == 1
        assert metrics.chunks_processed == 4

    def test_extract_efficiency_from_rag_response(self) -> None:
        resp = RAGResponse(
            query="Test query",
            answer="Test answer",
            top_k=3,
            retrieved_chunk_ids=["c1", "c2", "c3"],
            latency_ms=125.5,
            metadata={"retrieval_calls": 2, "escalation_count": 0},
        )
        eff = extract_efficiency_from_response(resp)
        assert eff.latency_ms == 125.5
        assert eff.chunks_processed == 3
        assert eff.retrieval_calls == 2
        assert eff.escalation_count == 0
        assert eff.token_usage is None  # Never fabricated when not provided

    def test_aggregate_efficiency_metrics(self) -> None:
        m1 = EfficiencyMetrics(
            latency_ms=100.0,
            retrieval_calls=1,
            chunks_processed=5,
            escalation_count=0,
            token_usage=LLMUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
        )
        m2 = EfficiencyMetrics(
            latency_ms=200.0,
            retrieval_calls=2,
            chunks_processed=10,
            escalation_count=1,
            token_usage=LLMUsage(prompt_tokens=100, completion_tokens=30, total_tokens=130),
        )

        agg = aggregate_efficiency_metrics([m1, m2])
        assert agg["count"] == 2
        assert agg["mean_latency_ms"] == 150.0
        assert agg["total_retrieval_calls"] == 3
        assert agg["mean_retrieval_calls"] == 1.5
        assert agg["total_chunks_processed"] == 15
        assert agg["total_escalations"] == 1
        assert agg["total_tokens"] == {
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "total_tokens": 200,
        }


# =============================================================================
# 8. MODEL SERIALIZATION TESTS
# =============================================================================


class TestModelSerialization:
    """Tests for full model serialization and round-trip deserialization."""

    def test_question_evaluation_result_serialization(self) -> None:
        result = QuestionEvaluationResult(
            question_id="rag_fact_01",
            query="What is the default scoring in GridSearchCV?",
            reference_answer="Uses default score method of underlying estimator.",
            generated_answer="GridSearchCV defaults to estimator's score method.",
            retrieved_chunk_ids=["chunk_001", "chunk_002"],
            ground_truth_chunk_ids=["chunk_001"],
            retrieval_metrics=RetrievalMetrics(
                recall_at_k={1: 1.0, 3: 1.0},
                mrr=1.0,
                ndcg_at_k={1: 1.0, 3: 1.0},
                hit_at_k={1: True, 3: True},
                retrieved_chunk_ids=["chunk_001", "chunk_002"],
                ground_truth_chunk_ids=["chunk_001"],
            ),
            answer_scores=AnswerQualityScores(
                correctness=4.0,
                faithfulness=4.0,
                relevance=4.0,
                overall=4.0,
                reason="Direct and correct.",
            ),
            efficiency=EfficiencyMetrics(
                latency_ms=120.5,
                retrieval_calls=1,
                chunks_processed=2,
                escalation_count=0,
                token_usage=LLMUsage(
                    prompt_tokens=250, completion_tokens=35, total_tokens=285
                ),
            ),
            routing=RoutingEvaluation(
                predicted_strategy="DENSE",
                expected_strategy="DENSE",
                strategy_match=True,
                predicted_intent="FACT_LOOKUP",
                expected_intent="FACT_LOOKUP",
                intent_match=True,
                predicted_complexity="SIMPLE",
                expected_complexity="SIMPLE",
                complexity_match=True,
                routing_reason="Single factual concept",
            ),
            metadata={"system_name": "adaq_rag_adaptive", "temperature": 0.0},
        )

        # 1. model_dump() -> dict
        d = result.model_dump()
        assert isinstance(d, dict)
        assert d["question_id"] == "rag_fact_01"
        assert d["retrieval_metrics"]["mrr"] == 1.0
        assert d["answer_scores"]["overall"] == 4.0
        assert d["routing"]["strategy_match"] is True

        # 2. model_dump_json() -> str
        json_str = result.model_dump_json()
        assert isinstance(json_str, str)
        assert "rag_fact_01" in json_str
        assert "adaq_rag_adaptive" in json_str

        # 3. Round-trip deserialization from dict
        reloaded_dict = QuestionEvaluationResult.model_validate(d)
        assert reloaded_dict.question_id == result.question_id
        assert reloaded_dict.retrieval_metrics.mrr == result.retrieval_metrics.mrr
        assert reloaded_dict.answer_scores.overall == result.answer_scores.overall
        assert reloaded_dict.efficiency.token_usage.total_tokens == 285

        # 4. Round-trip deserialization from JSON string
        reloaded_json = QuestionEvaluationResult.model_validate_json(json_str)
        assert reloaded_json.question_id == result.question_id
        assert reloaded_json.routing.expected_strategy == "DENSE"
        assert reloaded_json.efficiency.latency_ms == 120.5

    def test_model_serialization_with_none_token_usage(self) -> None:
        result = QuestionEvaluationResult(
            question_id="rag_fact_02",
            query="Test query",
            reference_answer="Ref",
            generated_answer="Gen",
            retrieval_metrics=RetrievalMetrics(),
            answer_scores=AnswerQualityScores(
                correctness=3.0, faithfulness=3.0, relevance=3.0
            ),
            efficiency=EfficiencyMetrics(
                latency_ms=50.0,
                token_usage=None,  # No tokens
            ),
            routing=None,
        )

        json_str = result.model_dump_json()
        reloaded = QuestionEvaluationResult.model_validate_json(json_str)
        assert reloaded.efficiency.token_usage is None
        assert reloaded.routing is None
