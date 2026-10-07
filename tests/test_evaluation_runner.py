"""Focused test suite for AdaQ-RAG Evaluation Runner (Part 2).

Tests cover:
- Benchmark loading and validation
- Question filtering and subset selection
- System name normalization and selection
- EvidenceItem to RetrievalResult adapter
- Harness execution across all 4 systems (mock mode)
- Result construction and serialization
- MockJudge integration and system isolation
- Error/failure handling during evaluation
- Aggregate metric computation
- Saving and reloading evaluation results
"""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from adaq_rag.llm.mock import MockLLMProvider
from evaluation.rag.judge import MockJudge
from evaluation.rag.models import (
    AnswerQualityScores,
    QuestionEvaluationResult,
)
from evaluation.rag.runner import (
    DEFAULT_BENCHMARK_PATH,
    SUPPORTED_SYSTEMS,
    SYSTEM_ADAPTIVE_RAG,
    SYSTEM_BASIC_RAG,
    SYSTEM_FULL_ADAQ_RAG,
    SYSTEM_HYBRID_RAG,
    EvaluationHarness,
    EvaluationRunner,
    adapt_evidence_to_retrieval_results,
    filter_benchmark,
    load_benchmark,
    normalize_system_name,
)
from adaq_rag.retrieval.evidence import EvidenceItem


# =============================================================================
# 1. BENCHMARK LOADING & FILTERING TESTS
# =============================================================================


class TestBenchmarkLoadingAndFiltering:
    """Tests for loading and filtering the 90-question benchmark dataset."""

    def test_load_benchmark_default(self) -> None:
        data = load_benchmark()
        assert isinstance(data, list)
        assert len(data) == 90

        sample = data[0]
        assert "question_id" in sample
        assert "query" in sample
        assert "expected_intent" in sample
        assert "expected_complexity" in sample
        assert "expected_strategy" in sample
        assert "ground_truth_chunk_ids" in sample
        assert "reference_answer" in sample

    def test_load_benchmark_custom_path(self, tmp_path: Path) -> None:
        p = tmp_path / "custom_bench.json"
        items = [{"question_id": "test_01", "query": "Test query"}]
        p.write_text(json.dumps(items), encoding="utf-8")

        loaded = load_benchmark(p)
        assert len(loaded) == 1
        assert loaded[0]["question_id"] == "test_01"

    def test_load_benchmark_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_benchmark(tmp_path / "non_existent.json")

    def test_filter_benchmark_by_ids(self) -> None:
        questions = [
            {"question_id": "q1", "query": "one"},
            {"question_id": "q2", "query": "two"},
            {"question_id": "q3", "query": "three"},
        ]
        filtered = filter_benchmark(questions, question_ids=["q1", "q3"])
        assert len(filtered) == 2
        assert [q["question_id"] for q in filtered] == ["q1", "q3"]

    def test_filter_benchmark_by_limit(self) -> None:
        questions = [
            {"question_id": "q1"},
            {"question_id": "q2"},
            {"question_id": "q3"},
        ]
        filtered = filter_benchmark(questions, limit=2)
        assert len(filtered) == 2
        assert [q["question_id"] for q in filtered] == ["q1", "q2"]


# =============================================================================
# 2. SYSTEM SELECTION & NORMALIZATION TESTS
# =============================================================================


class TestSystemSelection:
    """Tests for system name normalization and selection."""

    def test_normalize_system_name(self) -> None:
        assert normalize_system_name("basic") == SYSTEM_BASIC_RAG
        assert normalize_system_name("Basic RAG") == SYSTEM_BASIC_RAG
        assert normalize_system_name("baseline_rag") == SYSTEM_BASIC_RAG

        assert normalize_system_name("hybrid") == SYSTEM_HYBRID_RAG
        assert normalize_system_name("Hybrid RAG") == SYSTEM_HYBRID_RAG

        assert normalize_system_name("adaptive") == SYSTEM_ADAPTIVE_RAG
        assert normalize_system_name("Adaptive RAG") == SYSTEM_ADAPTIVE_RAG

        assert normalize_system_name("full") == SYSTEM_FULL_ADAQ_RAG
        assert normalize_system_name("Full AdaQ-RAG") == SYSTEM_FULL_ADAQ_RAG
        assert normalize_system_name("adaq_rag") == SYSTEM_FULL_ADAQ_RAG

    def test_normalize_system_name_invalid(self) -> None:
        with pytest.raises(ValueError):
            normalize_system_name("unknown_pipeline")


# =============================================================================
# 3. EVIDENCE ADAPTER TESTS
# =============================================================================


class TestEvidenceAdapter:
    """Tests for adapt_evidence_to_retrieval_results adapter."""

    def test_adapt_evidence_item_to_retrieval_result(self) -> None:
        ev = EvidenceItem(
            chunk_id="chunk_101",
            content="Evidence content text",
            doc_id="doc_linear",
            doc_title="Linear Models",
            section_title="Ridge",
            section_level=2,
            initial_score=0.75,
            final_rerank_score=0.92,
            metadata={"source_url": "https://example.com"},
        )

        results = adapt_evidence_to_retrieval_results([ev])
        assert len(results) == 1
        res = results[0]

        assert res.chunk_id == "chunk_101"
        assert res.score == 0.92  # final_rerank_score mapped to score
        assert res.content == "Evidence content text"
        assert res.doc_id == "doc_linear"
        assert res.doc_title == "Linear Models"
        assert res.section_title == "Ridge"
        assert res.section_level == 2
        assert res.rerank_score == 0.92

    def test_adapt_evidence_fallback_initial_score(self) -> None:
        ev = EvidenceItem(
            chunk_id="chunk_102",
            content="Content",
            initial_score=0.65,
            final_rerank_score=None,
        )
        results = adapt_evidence_to_retrieval_results([ev])
        assert results[0].score == 0.65


# =============================================================================
# 4. HARNESS & ALL 4 SYSTEMS EXECUTION (MOCK MODE)
# =============================================================================


class TestHarnessExecution:
    """Tests executing all 4 systems using Mock Harness."""

    @pytest.fixture
    def mock_harness(self) -> EvaluationHarness:
        judge = MockJudge(
            default_scores=AnswerQualityScores(
                correctness=3.5,
                faithfulness=3.5,
                relevance=3.5,
                overall=3.5,
                reason="Mock judge score",
            )
        )
        return EvaluationHarness.create_mock(judge=judge)

    @pytest.fixture
    def sample_question(self) -> dict:
        return {
            "question_id": "test_fact_01",
            "query": "What is the default scoring strategy in GridSearchCV if scoring is None?",
            "expected_intent": "FACT_LOOKUP",
            "expected_complexity": "SIMPLE",
            "expected_strategy": "DENSE",
            "ground_truth_chunk_ids": ["mock_chunk_1"],
            "reference_answer": "It uses the estimator default score method.",
        }

    def test_run_basic_rag(self, mock_harness: EvaluationHarness, sample_question: dict) -> None:
        res = mock_harness.run_basic_rag(sample_question)
        assert isinstance(res, QuestionEvaluationResult)
        assert res.question_id == "test_fact_01"
        assert res.metadata["system_name"] == SYSTEM_BASIC_RAG
        assert res.efficiency.retrieval_calls == 1
        assert res.routing.predicted_strategy == "DENSE"
        assert res.routing.strategy_match is True
        assert res.retrieval_metrics.mrr > 0.0
        assert res.answer_scores.overall == 3.5

    def test_run_hybrid_rag(self, mock_harness: EvaluationHarness, sample_question: dict) -> None:
        res = mock_harness.run_hybrid_rag(sample_question)
        assert isinstance(res, QuestionEvaluationResult)
        assert res.question_id == "test_fact_01"
        assert res.metadata["system_name"] == SYSTEM_HYBRID_RAG
        assert res.efficiency.retrieval_calls == 2
        assert res.routing.predicted_strategy == "HYBRID"
        assert res.answer_scores.overall == 3.5

    def test_run_adaptive_rag(self, mock_harness: EvaluationHarness, sample_question: dict) -> None:
        res = mock_harness.run_adaptive_rag(sample_question)
        assert isinstance(res, QuestionEvaluationResult)
        assert res.question_id == "test_fact_01"
        assert res.metadata["system_name"] == SYSTEM_ADAPTIVE_RAG
        assert res.routing.predicted_strategy in ("DENSE", "HYBRID", "MULTI_STEP")
        assert res.answer_scores.overall == 3.5

    def test_run_full_adaq_rag(self, mock_harness: EvaluationHarness, sample_question: dict) -> None:
        res = mock_harness.run_full_adaq_rag(sample_question)
        assert isinstance(res, QuestionEvaluationResult)
        assert res.question_id == "test_fact_01"
        assert res.metadata["system_name"] == SYSTEM_FULL_ADAQ_RAG
        assert res.routing.predicted_strategy in ("DENSE", "HYBRID", "MULTI_STEP")
        assert res.answer_scores.overall == 3.5

    def test_dispatch_run_system(self, mock_harness: EvaluationHarness, sample_question: dict) -> None:
        for sys_name in SUPPORTED_SYSTEMS:
            res = mock_harness.run_system(sys_name, sample_question)
            assert res.metadata["system_name"] == sys_name


# =============================================================================
# 5. RUNNER SUBSET EXECUTION & AGGREGATION TESTS
# =============================================================================


class TestRunnerExecutionAndAggregation:
    """Tests EvaluationRunner executing subsets and computing aggregate metrics."""

    def test_runner_evaluates_subset(self) -> None:
        judge = MockJudge()
        harness = EvaluationHarness.create_mock(judge=judge)
        runner = EvaluationRunner(harness=harness, judge=judge)

        questions = [
            {
                "question_id": "q1",
                "query": "What is Ridge?",
                "expected_strategy": "DENSE",
                "ground_truth_chunk_ids": ["mock_chunk_1"],
                "reference_answer": "Ridge uses L2 norm.",
            },
            {
                "question_id": "q2",
                "query": "Compare Ridge and Lasso.",
                "expected_strategy": "HYBRID",
                "ground_truth_chunk_ids": ["mock_chunk_2"],
                "reference_answer": "Ridge is L2, Lasso is L1.",
            },
        ]

        # Run 2 questions on 2 systems (basic_rag, adaptive_rag)
        results = runner.evaluate_questions(
            questions=questions,
            systems=["basic_rag", "adaptive_rag"],
        )

        assert len(results) == 4  # 2 questions * 2 systems

        # Compute summary
        summary = runner.aggregate_results(results)
        assert SYSTEM_BASIC_RAG in summary
        assert SYSTEM_ADAPTIVE_RAG in summary

        basic_summary = summary[SYSTEM_BASIC_RAG]
        assert basic_summary["question_count"] == 2
        assert "retrieval" in basic_summary
        assert "routing" in basic_summary
        assert "efficiency" in basic_summary
        assert "quality" in basic_summary
        assert basic_summary["quality"]["mean_overall"] > 0.0

    def test_runner_failure_handling(self) -> None:
        judge = MockJudge()
        harness = EvaluationHarness.create_mock(judge=judge)
        # Mock run_system to fail for one system
        original_run = harness.run_system

        def flappy_run(sys_name: str, item: dict):
            if sys_name == SYSTEM_BASIC_RAG:
                raise RuntimeError("Simulated pipeline failure")
            return original_run(sys_name, item)

        harness.run_system = flappy_run

        runner = EvaluationRunner(harness=harness, judge=judge)
        questions = [{"question_id": "q_fail", "query": "Test query"}]

        results = runner.evaluate_questions(questions, systems=["basic_rag", "hybrid_rag"])
        assert len(results) == 2

        # Basic RAG should be caught with error metadata
        basic_res = next(r for r in results if r.metadata.get("system_name") == SYSTEM_BASIC_RAG)
        assert "error" in basic_res.metadata
        assert "Simulated pipeline failure" in basic_res.metadata["error"]
        assert basic_res.answer_scores.overall == 0.0

        # Hybrid RAG should have succeeded normally
        hybrid_res = next(r for r in results if r.metadata.get("system_name") == SYSTEM_HYBRID_RAG)
        assert "error" not in hybrid_res.metadata
        assert hybrid_res.answer_scores.overall > 0.0

    def test_save_and_reload_results(self, tmp_path: Path) -> None:
        judge = MockJudge()
        harness = EvaluationHarness.create_mock(judge=judge)
        runner = EvaluationRunner(harness=harness, judge=judge)

        questions = [
            {
                "question_id": "q_save",
                "query": "Sample query",
                "ground_truth_chunk_ids": ["mock_chunk_1"],
                "reference_answer": "Ref ans",
            }
        ]
        results = runner.evaluate_questions(questions, systems=["basic_rag"])

        out_file = tmp_path / "eval_results.json"
        runner.save_results(results, out_file)

        assert out_file.is_file()
        with open(out_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["question_id"] == "q_save"
        assert data[0]["metadata"]["system_name"] == SYSTEM_BASIC_RAG
